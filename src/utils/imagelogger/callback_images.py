from typing import Optional
import torch
import torchvision
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid
from PIL import Image
from io import BytesIO
import numpy as np  # utile pour flatten les axes si besoin

from src.utils.imagelogger.image_loader import get_image_logger

class LogPredictionsCallback(Callback):
    def __init__(self, num_images: int = 8, image_log_dir: Optional[str] = None):
        super().__init__()
        self.num_images = num_images
        self.image_loggers = []
        self.object_dict = None
        self.image_log_dir = image_log_dir

    def set_object_dict(self, object_dict):
        self.object_dict = object_dict
    
    def set_path_dir(self, image_log_dir):
        self.image_log_dir = image_log_dir

    def _init_loggers(self, trainer):
        loggers = self.object_dict["logger"]
        loggers = loggers 

        logger_instances = []
        
        for l in loggers:
            logger_type = type(l).__name__
            if logger_type == "TensorBoardLogger":
                print("✅ TensorBoardLogger détecté")
                logger_instances.append(get_image_logger("tensorboard", writer=l.experiment)) 
                self.tensorboard_log_dir = l.log_dir
            elif logger_type == "MLFlowLogger":
                print("✅ MLFlowLogger détecté")
                logger_instances.append(get_image_logger("mlflow", run_id=l.run_id, local_dir=self.image_log_dir))
            else:
                print(f"⚠️ Logger non pris en charge : {type(l)}")

        return logger_instances

    

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.image_loggers:
            self.image_loggers = self._init_loggers(trainer)

        val_loader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_loader))
        images, labels = batch[:2]
        images, labels = images[:self.num_images], labels[:self.num_images]
        images = images.to(pl_module.device)

        pl_module.eval()
        with torch.no_grad():
            outputs = pl_module(images)
            preds = torch.argmax(outputs[1] if isinstance(outputs, tuple) else outputs, dim=1)
        pl_module.train()

        # Image grid
        grid_raw = make_grid(images.cpu(), nrow=4, normalize=True)

        # Annotated predictions
        cols = min(self.num_images, 4)
        rows = (self.num_images + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]

        for i in range(self.num_images):
            img = images[i].cpu().permute(1, 2, 0).squeeze()
            axs[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axs[i].set_title(f"Label: {labels[i].item()} / Pred: {preds[i].item()}")
            axs[i].axis('off')

        for j in range(self.num_images, len(axs)):
            axs[j].axis('off')

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_pil = Image.open(buf)
        plt.close(fig)

        # Logging
        for image_logger in self.image_loggers:
            image_logger.log_image(grid_raw, "Validation/Raw Images", trainer.current_epoch)
            image_logger.log_image(img_pil, "Validation/Predictions Annotated", trainer.current_epoch)

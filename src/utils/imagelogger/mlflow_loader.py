import mlflow
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from src.utils.imagelogger.image_loggerInterface import ImageLogger
import torchvision.transforms as T
from mlflow.tracking import MlflowClient

class MLflowImageLogger(ImageLogger):
    def __init__(self, run_id):
        self.run_id = run_id

    def log_image(self, image, tag: str, step: int):
        # Si image est un Tensor, on le convertit en image PIL
        if hasattr(image, "cpu"):
            image = image.cpu()
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = T.ToPILImage()(image)
            elif image.ndim == 2:
                image = T.ToPILImage()(image.unsqueeze(0))
            else:
                raise ValueError("Unsupported image tensor shape")

        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        elif isinstance(image, Image.Image):
            pass  # déjà OK

        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Save to buffer
        buf = BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

     
         # Convertir le buffer en image PIL
        pil_img = Image.open(buf)
        
        
        client = MlflowClient()
        client.log_image(run_id=self.run_id, image=pil_img, artifact_file=f"{tag.replace('/', '_')}_{step}.png")



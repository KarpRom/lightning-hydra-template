import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from src.utils.imagelogger.image_loggerInterface import ImageLogger
from mlflow.tracking import MlflowClient

class MLflowImageLogger(ImageLogger):
    def __init__(self, run_id, local_dir: str):
        self.run_id = run_id
        self.local_dir = local_dir
        os.makedirs(self.local_dir, exist_ok=True)
        self.client = MlflowClient()

    def log_image(self, image, tag: str, step: int):
        # Convertir en PIL
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
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Sauvegarde dans le dossier partagé
        filename = f"{tag.replace('/', '_')}_{step}.png"
        local_path = os.path.join(self.local_dir, filename)
        image.save(local_path)

        # Utilise log_artifact pour référencer le fichier dans le bon chemin
        self.client.log_artifact(self.run_id, local_path, artifact_path="images")

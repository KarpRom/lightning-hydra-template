import torchvision
from torchvision.transforms.functional import to_tensor
from PIL import Image
import torch

from src.utils.imagelogger.image_loggerInterface import ImageLogger
from torchvision.transforms import ToTensor

class TensorBoardImageLogger(ImageLogger):
    def __init__(self, writer):
        self.writer = writer

    def log_image(self, image, tag: str, step: int):
        if isinstance(image, torch.Tensor):
            self.writer.add_image(tag, image, global_step=step)
        elif isinstance(image, Image.Image):  # PIL.Image
            tensor_img = ToTensor()(image)
            self.writer.add_image(tag, tensor_img, global_step=step)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")


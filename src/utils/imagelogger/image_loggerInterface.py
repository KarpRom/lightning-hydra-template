from abc import ABC, abstractmethod
from typing import Any


class ImageLogger(ABC):
    @abstractmethod
    def log_image(self, tag: str, image: Any, step: int):
        """Log an image to the logging backend."""
        pass

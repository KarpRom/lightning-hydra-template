
from src.utils.imagelogger.image_loggerInterface import ImageLogger
from src.utils.imagelogger.tensorboard_loader import TensorBoardImageLogger
from src.utils.imagelogger.mlflow_loader import MLflowImageLogger

def get_image_logger(logger_type: str, **kwargs):
    if logger_type == "tensorboard":
        return TensorBoardImageLogger(writer=kwargs.get("writer"))
    elif logger_type == "mlflow":
        return MLflowImageLogger(run_id=kwargs.get("run_id"))
    else:
        raise ValueError(f"Unsupported logger type: {logger_type}")

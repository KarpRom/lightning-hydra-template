import os
from pathlib import Path

import pytest
import torch

from src.data.mnist_datamodule import MNISTDataModule

@pytest.mark.parametrize("batch_size", [32, 128])
def test_dataset(batch_size: int) -> None:
    """Tests your Dataset to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match. 

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    #raise NotImplementedError  #delete this if you implement the test

    

    # Résolution manuelle des chemins
    root_dir = os.environ["PROJECT_ROOT"]
    data_dir = os.path.join(root_dir, "data")
    

    # Convertit en dict pour instanciation
    dm = MNISTDataModule(data_dir=data_dir,batch_size=batch_size)

    dm.prepare_data()
    dm.setup()

    # Vérifie la présence des datasets
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    # Vérifie un batch
    batch = next(iter(dm.train_dataloader()))
    x, y = batch[:2]

    # Vérifie le type et la taille de x et y 
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert isinstance(x, None)
    assert isinstance(y, None)


from pathlib import Path

import pytest
import torch

@pytest.mark.parametrize("batch_size", [32, 128])
def test_dataset(batch_size: int) -> None:
    """Tests your Dataset to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    raise NotImplementedError  #delete this if you implement the test

    data_dir = "data/"

    dm = null #your Datamodule
    dm.prepare_data()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == null #x type ,use torch.
    assert y.dtype == null #y type


import re

import pytest
import torch

from torchtime.lightning import UEADataModule

DATASET = "ArrowHead"
SEED = 456789


class TestLightningUEAArrowHead:
    """Test UEA class with ArrowHead data set."""

    def test_lightning_datamodule_setup(self):

        data = UEADataModule(
            dataset_name=DATASET,
            train_val_ratio=(0.9, 0.1),
            batch_size=25,
            seed=SEED,
            append_time_as_first_dim=False,
        )
        data.prepare_data()
        data.setup("")
        batch = next(iter(data.train_dataloader()))

        assert batch[0].shape == torch.Size([25, 1, 251])
        assert batch[1].shape == torch.Size([25])
        assert batch[2].shape == torch.Size([25])

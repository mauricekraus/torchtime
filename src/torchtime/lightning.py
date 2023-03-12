import torch
from torch.utils.data import DataLoader, TensorDataset
from torchtime.data import UEA

try:
    from lightning import LightningDataModule

except ModuleNotFoundError as err:
    # TODO: HOW can we do that on a package level?
    print(
        "\033[31mError: Lightning is not installed. Please run `pip install lightning` to install the package and try again.\033[0m"
    )
    exit(0)


class UEADataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        train_val_ratio: tuple[float, float],
        batch_size: int,
        seed: int,
        append_time_as_first_dim: bool = False,
    ) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.seed = seed
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.append_time_as_first_dim = append_time_as_first_dim

    def prepare_data(self) -> None:

        # doanloading the data
        _ = UEA(
            dataset=self.dataset_name,
            train_prop=0.9,
            split="train",
            seed=self.seed,
            path="/tmp/datasets/",
        )

    def setup(self, stage: str) -> None:

        self.data = UEA(
            dataset=self.dataset_name,
            split="train",
            train_prop=self.train_val_ratio[0],
            val_prop=self.train_val_ratio[1],
            seed=self.seed,
            time=self.append_time_as_first_dim,
            path="/tmp/datasets/",
            # one_hot_y=False,
        )

        if stage == "test":
            # Test data
            self.test_data = TensorDataset(
                self.data.X_test,  # type: ignore
                self.data.y_test,  # type: ignore
                self.data.length_test,  # type: ignore
            )
        else:
            # Validation data
            self.val_data = TensorDataset(
                self.data.X_val,  # type: ignore
                self.data.y_val,  # type: ignore
                self.data.length_val,  # type: ignore
            )

    def convert_to_single_y(self, batch):
        xs = []
        ys = []
        lengths = []
        if isinstance(batch[0], dict):
            batch = [(b["X"], b["y"], b["length"]) for b in batch]
        for b in batch:
            xs.append(b[0])
            ys.append(b[1].max(dim=0, keepdim=True).indices)
            lengths.append(b[2])
        return torch.stack(xs), torch.stack(ys).squeeze(-1), torch.stack(lengths)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.convert_to_single_y,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            collate_fn=self.convert_to_single_y,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=self.convert_to_single_y,
        )

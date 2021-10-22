import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


from src.load_data import load_train_data, load_test_data, load_train_labels, load_test_labels


class MNIST(Dataset):

    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,) -> None:
        if len(data) != len(labels):
            raise ValueError('data and labels must be the same length. '
                             f'{len(data)} != {len(labels)}')
        self.data = torch.from_numpy(data).to(torch.float32)
        self.labels = torch.from_numpy(labels).to(torch.long)


    def __getitem__(self, index: int) -> T_co:
        image = self.data[index]
        label = self.labels[index]
        return (image, label)

    def __len__(self):
        return len(self.data)


def get_train_dataloader(
        batch_size: int,
        transforms=None) -> DataLoader:
    return DataLoader(
        # MNIST(load_train_data(), load_train_labels(), transform=transforms),
        MNIST(load_train_data(), load_train_labels()),
        batch_size=batch_size,
        shuffle=True)


def get_test_dataloader(
        batch_size: int,
        transforms=None) -> DataLoader:
    return DataLoader(
        # MNIST(load_test_data(), load_test_labels(), transform=transforms),
        MNIST(load_test_data(), load_test_labels()),
        batch_size=batch_size,
        shuffle=False)

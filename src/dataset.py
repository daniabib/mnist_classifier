import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


from src.load_data import load_train_data, load_test_data, load_train_labels, load_test_labels


class MNIST(Dataset):

    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 transform=None) -> None:
        if len(data) != len(labels):
            raise ValueError('data and labels must be the same length. '
                             f'{len(data)} != {len(labels)}')
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index: int) -> T_co:
        image = self.data[index]
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return (image, label)

    def __len__(self):
        return len(self.data)


def get_train_dataloader(
        batch_size: int,
        transforms=None) -> DataLoader:
    return DataLoader(
        MNIST(load_train_data(), load_train_labels(), transform=transforms),
        batch_size=batch_size,
        shuffle=True)


def get_test_dataloader(
        batch_size: int,
        transforms=None) -> DataLoader:
    return DataLoader(
        MNIST(load_test_data(), load_test_labels(), transform=transforms),
        batch_size=batch_size,
        shuffle=False)

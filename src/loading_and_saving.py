#!/usr/bin/env python3

import numpy as np
import torch
import h5py
import pathlib
from torch.utils.data import Dataset, DataLoader
from util_functions import plot_image, plot_x_y
from .configuration import ROOT_PATH
import pandas as pd

from typing import List


class DoubleDataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')
        self.x = self.file['x']
        self.y = self.file['y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


class SingleDataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')
        self.x = self.file['x']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])


DATA_FILE = ROOT_PATH / "data/torch/train_dataset_50_50.h5"
UPSCALE_DATA_FILE = ROOT_PATH / "data/torch/train_dataset_upscaled.h5"
TEST_DATA_FILE = ROOT_PATH / "data/torch/test_dataset.h5"
TEST_UPSCALE_DATA_FILE = ROOT_PATH / "data/torch/test_dataset_upscaled.h5"


def load_original_csvs() -> List:
    """
    Loads the original data from the csv file and returns a list of the data
    """
    x = pd.read_csv(ROOT_PATH / "data/original/train_damaged.csv")
    y = pd.read_csv(ROOT_PATH / "data/original/train_undamaged.csv")
    z = pd.read_csv(ROOT_PATH / "data/original/test_damaged.csv")
    return [x, y, z]


def load_train_data(batch_size: int = 5, upscaled=False, shuffle=True) -> DataLoader:
    filepath = UPSCALE_DATA_FILE if upscaled else DATA_FILE
    dataset = DoubleDataset(filepath)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_test_data(batch_size: int = 5, upscaled=False, shuffle=False) -> DataLoader:
    filepath = TEST_UPSCALE_DATA_FILE if upscaled else TEST_DATA_FILE
    dataset = SingleDataset(filepath)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def to_numpy(df: pd.DataFrame, size=50) -> np.ndarray:
    arr = df.copy()
    arr.index = arr.pop("Id")
    arr = arr.to_numpy().reshape(-1, size, size)
    return arr


def from_numpy(arr: np.ndarray) -> pd.DataFrame:
    y = arr.reshape(arr.shape[0], -1)
    y_df = pd.DataFrame(y)

    y_df.index.name = "Id"
    return y_df


def to_file(x: torch.Tensor, name="submission.csv") -> None:
    df = from_numpy(x.detach().numpy())
    df.to_csv(ROOT_PATH / name)


if __name__ == "__main__":
    x, y, z = load_original_csvs()
    loader = load_train_data(500, shuffle=False)
    for i, data in enumerate(loader):
        inputs, labels = data
        to_file(inputs)

#!/usr/bin/env python3

import numpy as np
import torch
import h5py
import pandas as pd

def load() -> List:
    """
    Loads the original data from the csv file and returns a list of the data
    """
    x = pd.read_csv("data/original/train_damaged.csv")
    y = pd.read_csv("data/original/train_undamaged.csv")
    z = pd.read_csv("data/original/test_damaged.csv")
    return [x, y, z]


def to_numpy(df: pd.DataFrame, size=50) -> np.ndarray:
    arr = df.copy()
    arr.index = arr.pop("Id")
    arr = arr.to_numpy().reshape(-1, size, size)
    return arr



def upscale_damage(arr: np.ndarray) -> np.ndarray:
    """
    Upscales the damaged images to the same size as the undamaged images
    by repeating the pixels (1 pixel becomes 4 pixels)
    """
    upscaled = np.copy(arr)
    upscaled = np.repeat(upscaled, 2, axis=1)
    upscaled = np.repeat(upscaled, 2, axis=2)
    return upscaled


def save_as_numpy(arr: np.ndarray, name: str) -> None:
    """
    Saves the numpy array as a numpy file
    """
    np.save(f"data/numpy/{name}.npy", arr)


def save_as_tensor(arr: np.ndarray, name: str) -> None:
    """
    Saves the numpy array as a pytorch tensor
    """
    torch.save(torch.from_numpy(arr), f"data/torch/{name}.pt")


def save_for_dataloader(x: np.ndarray, y: np.ndarray, name: str) -> None:
    """
    Saves the numpy array as a pytorch tensor
    """
    with h5py.File(f'data/torch/{name}.h5', 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)


def main() -> None:
    x, y, z = load()

    np_x = to_numpy(x)
    np_y = to_numpy(y, 100)
    np_z = to_numpy(z)

    up_x = upscale_damage(np_x)
    up_z = upscale_damage(np_z)

    numpy = True
    if numpy:
        save_as_numpy(np_x, "train_damaged")
        save_as_numpy(np_y, "train_undamaged")
        save_as_numpy(np_z, "test_damaged")
        save_as_numpy(up_x, "train_damaged_upscaled")
        save_as_numpy(up_z, "test_damaged_upscaled")

    tensor = True
    if tensor:
        save_as_tensor(np_x, "train_damaged")
        save_as_tensor(np_y, "train_undamaged")
        save_as_tensor(np_z, "test_damaged")
        save_as_tensor(up_x, "train_damaged_upscaled")
        save_as_tensor(up_z, "test_damaged_upscaled")

    dataloader = True
    if dataloader:
        save_for_dataloader(np_x, np_y, "train_dataset_50_50")
        save_for_dataloader(up_x, np_y, "train_dataset_upscaled")


if __name__ == "__main__":
    main()

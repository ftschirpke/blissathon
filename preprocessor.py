#!/usr/bin/env python3

import numpy as np
import torch
import h5py
import pathlib
from torch.utils.data import Dataset, DataLoader
from util_functions import plot_image, plot_x_y
import pandas as pd
from typing import List


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


def neighbors(arr: np.ndarray, i: int, j: int):
    neighbors = []
    if i > 0 and arr[i - 1][j] != 0:
        yield (i - 1, j)
    if i < 49 and arr[i + 1][j] != 0:
        yield (i + 1, j)
    if j > 0 and arr[i][j - 1] != 0:
        yield (i, j - 1)
    if j < 49 and arr[i][j + 1] != 0:
        yield (i, j + 1)
    if i > 0 and j > 0 and arr[i - 1][j - 1] != 0:
        yield (i - 1, j - 1)
    if i > 0 and j < 49 and arr[i - 1][j + 1] != 0:
        yield (i - 1, j + 1)
    if i < 49 and j > 0 and arr[i + 1][j - 1] != 0:
        yield (i + 1, j - 1)
    if i < 49 and j < 49 and arr[i + 1][j + 1] != 0:
        yield (i + 1, j + 1)
    return neighbors


def main() -> None:
    train_orig, _, test_orig = load()
    train_x = to_numpy(train_orig).squeeze()
    test_x = to_numpy(test_orig).squeeze()
    print("train", train_x.shape)
    print("test", test_x.shape)

    riss = to_numpy(pd.read_csv("rissv1.csv")).squeeze()
    r = riss.copy()

    print(riss.shape)

    count_riss = 0
    count_other = 0

    apply_order = []
    todo = []

    for i in range(len(r)):
        for j in range(len(r[i])):
            if r[i][j] == 0:
                print(i, j)
                count_riss += 1
                riss_neighbors = len(list(neighbors(r, i, j)))
                if riss_neighbors >= 4:
                    r[i][j] = 1
                    apply_order.append((i, j))
                else:
                    todo.append((i, j))
            else:
                count_other += 1
    print(count_riss, count_other)

    infinite_preventer = 0
    while len(todo) > 0:
        i, j = todo.pop(0)
        riss_neighbors = len(list(neighbors(r, i, j)))
        if riss_neighbors >= 4:
            r[i][j] = 1
            apply_order.append((i, j))
        else:
            todo.append((i, j))
        infinite_preventer += 1
        if infinite_preventer > 100000:
            print("infinite loop")
            return

    for image in test_x:
        for i, j in apply_order:
            n = list(neighbors(image, i, j))
            image[i][j] = np.mean(n)



if __name__ == "__main__":
    main()

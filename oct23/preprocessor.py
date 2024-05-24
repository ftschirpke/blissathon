#!/usr/bin/env python3

import numpy as np
import torch
from copy import deepcopy
import h5py
import pathlib
from torch.utils.data import Dataset, DataLoader
from util_functions import plot_image, plot_x_y
from data_transformer import load, to_numpy, save_for_dataloader
import pandas as pd
from typing import List


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


def upscale_damage(arr: np.ndarray) -> np.ndarray:
    """
    Upscales the damaged images to the same size as the undamaged images
    by repeating the pixels (1 pixel becomes 4 pixels)
    """
    upscaled = np.copy(arr)
    upscaled = np.repeat(upscaled, 2, axis=1)
    upscaled = np.repeat(upscaled, 2, axis=2)
    return upscaled


def main() -> None:
    train_orig, train_y_orig, test_orig = load()
    train_x = to_numpy(train_orig).squeeze()
    test_x = to_numpy(test_orig).squeeze()
    print("train", train_x.shape)
    print("test", test_x.shape)

    crack = to_numpy(pd.read_csv("crackv1.csv")).squeeze()
    r = deepcopy(crack)

    print(crack.shape)

    count_crack = 0
    count_other = 0

    apply_order = []
    todo = []

    for i in range(len(r)):
        for j in range(len(r[i])):
            if r[i][j] == 0:
                count_crack += 1
                crack_neighbors = len(list(neighbors(r, i, j)))
                if crack_neighbors >= 4:
                    r[i][j] = 1
                    apply_order.append((i, j))
                else:
                    todo.append((i, j))
            else:
                count_other += 1
    print(count_crack, count_other)

    infinite_preventer = 0
    while len(todo) > 0:
        i, j = todo.pop(0)
        crack_neighbors = len(list(neighbors(r, i, j)))
        if crack_neighbors >= 4:
            r[i][j] = 1
            apply_order.append((i, j))
        else:
            todo.append((i, j))
        infinite_preventer += 1
        if infinite_preventer > 100000:
            print("infinite loop")
            return

    print("apply order", len(apply_order), apply_order[:10], "...")

    better_train_x = deepcopy(train_x)
    r = deepcopy(crack)
    show = 0
    for i, image in enumerate(better_train_x):
        for i, j in apply_order:
            n = [image[x][y] for x, y in neighbors(r, i, j)]
            r[i][j] = 1
            image[i][j] = sum(n) / len(n)

        if show < 3:
            plot_x_y(image, train_x[i], f"train {i} without crack (maybe)")
            show += 1

    train_x_for_data = upscale_damage(better_train_x)
    train_y_for_data = to_numpy(train_y_orig, size=100)
    print("IMPORTANT:" + str(train_y_for_data.shape))
    save_for_dataloader(train_x_for_data, train_y_for_data, "train_dataset_100_100_without_crack")
    print(better_train_x.shape)

    better_test_x = deepcopy(test_x)
    r = deepcopy(crack)
    show = 0
    for i, image in enumerate(better_test_x):
        for i, j in apply_order:
            n = [image[x][y] for x, y in neighbors(r, i, j)]
            r[i][j] = 1
            image[i][j] = sum(n) / len(n)

        if show < 3:
            plot_x_y(image, test_x[i], f"test {i} without crack (maybe)")
            show += 1

    test_x_for_data = upscale_damage(better_test_x)
    with h5py.File('data/torch/test_dataset_100_100_without_crack.h5', 'w') as f:
        f.create_dataset('x', data=test_x_for_data)
    print(better_test_x.shape)


if __name__ == "__main__":
    main()

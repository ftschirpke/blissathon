#!/usr/bin/env python3

import numpy as np
import torch
import h5py
import pandas as pd
from util_functions import plot_image, plot_x_y
from loading_and_saving import load_train_data, to_file


    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
train_dataset = DoubleDataset("data/torch/train_dataset_50_50.h5")
test_dataset = SingleDataset("data/torch/test_dataset.h5")


if __name__ == "__main__":
    batch_size = 5
    dataloader = load_train_data(batch_size=batch_size)
    first = True
    first_x = None
    first_y = None
    prev_x = None
    prev_y = None
    x = None
    y = None
    diff = torch.zeros((50, 50))
    for i, data in enumerate(dataloader):
        inputs, labels = data
        for j in range(batch_size):
            prev_x = x
            prev_y = y
            x, y = inputs[j], labels[j]
            if first:
                first_x = x
                first_y = y
                first = False
            else:
                for row_num, row in enumerate(x):
                    for col_num, col in enumerate(row):
                        diff[row_num][col_num] += abs(prev_x[row_num][col_num] - x[row_num][col_num])

    normed = torch.nn.functional.normalize(diff, p=2.0)
    normed[normed >= 0.05] = 1
    normed[normed < 0.05] = 0
    to_file(normed, "rissv1.csv")
    print(normed[:15][:15])
    plot_image(normed, "diff")

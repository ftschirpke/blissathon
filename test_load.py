#!/usr/bin/env python3

import numpy as np
import torch
import h5py
import pandas as pd
from util_functions import plot_image, plot_x_y
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')
        self.x = self.file['x']
        self.y = self.file['y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


# Specify the path to your HDF5 file
data_file = 'data/torch/test_submission.h5'

batch_size = 5

# Create a DataLoader
dataset = CustomDataset(data_file)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# You can now use 'dataloader' to iterate through your data in batches
for i, data in enumerate(dataloader):
    inputs, labels = data
    x, y = inputs[0], labels[0]
    plot_x_y(x, y)
    break

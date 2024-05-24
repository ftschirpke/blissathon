#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import h5py
import pandas as pd
from src.loading_and_saving import load_test_data


class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            #            nn.Upsample(scale_factor=2),
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            #            nn.Upsample(scale_factor=2),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, 4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
            #            nn.Upsample(scale_factor=2),
        )
        self.decoder = nn.Sequential(
            #            nn.MaxPool2d(2, stride=2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 16, 2, padding=1),
            nn.ReLU(),
            #            nn.MaxPool2d(2, stride=2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            #            nn.MaxPool2d(2, stride=2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 1, 3, padding=1),
            # nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.decoder(x)
        return x


model = torch.load("/home/friedrich/Downloads/winninglatenocap.pth", map_location=torch.device('cpu'))
model.eval()

test_dataloader = load_test_data(batch_size=5, upscaled=True, shuffle=False)

predictions = []
for i, data in enumerate(test_dataloader, 0):
    data = data.cpu()
    data = torch.unsqueeze(data, dim=1).float()
    pred = model(data).cpu()

    arr = pred.detach().numpy()
    predictions.append(arr)

res = np.concatenate(predictions, axis=0)
print(res.shape)
res = res.reshape(res.shape[0], -1)
print(res.shape)
df = pd.DataFrame(res)
df.index.name = "Id"
df.to_csv("submission.csv")

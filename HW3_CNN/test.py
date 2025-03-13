import os
from preprocess import test_tfm, FoodDataset
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

class Conv(nn.Module):
    def __init__(self, intput_dim, output_dim, diff = True):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(intput_dim, output_dim, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )
        self.downSample = nn.Sequential(
            nn.Conv2d(intput_dim, output_dim, 3, 1, 1),
            nn.BatchNorm2d(output_dim)
        )
        self.diff = diff
        
    def forward(self, x):
        identify = x
        output = self.model(x)
        if(self.diff):
            identify = self.downSample(identify)
        return output + identify


class Classifier(nn.Module):

    def __init__(self):
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        super().__init__()
        
        self.cnn = nn.Sequential(
            Conv(3, 64),         # (64, 128, 128)
            Conv(64, 64, False), # (64, 128, 128)
            nn.MaxPool2d(2, 2),  # (64, 64, 64)
            Conv(64, 128),
            Conv(128, 128, False),
            nn.MaxPool2d(2, 2),
            Conv(128, 256),
            Conv(256, 256, False),
            nn.MaxPool2d(2, 2),  
            Conv(256, 512),
            Conv(512, 512, False),
            nn.MaxPool2d(2, 2),     
            Conv(512, 512, False),
            nn.MaxPool2d(2, 2),   #(512, 4, 4)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11),
        )
    def forward(self, x):
        out = self.cnn(x)
        
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


device = "cuda:3" if torch.cuda.is_available() else "cpu"
model = Classifier().to(device)
model.load_state_dict(torch.load('model_weights.pth'))

model.eval()
"cuda:3" if torch.cuda.is_available() else "cpu"
path = './ml2023spring-hw3/test/'

test_set = FoodDataset(path, tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)


prediction = []
with torch.no_grad():
    for data,_ in tqdm(test_loader):
        test_pred = model(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()

df = pd.DataFrame()
df["Id"] = [f"{i:>04}" for i in range(len(test_set))]
df["Category"] = prediction
df.to_csv("submission.csv", index = False)

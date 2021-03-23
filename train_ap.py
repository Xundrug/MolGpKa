from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import logging

import numpy as np
import os.path as osp
import pandas as pd

from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import DataLoader

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = Linear(2048, 1024)
        self.fc2 = Linear(1024, 516)
        self.fc3 = Linear(516, 256)
        self.fc4 = Linear(256, 128)
        self.fc5 = Linear(128, 1)


    def forward(self, data):
        x = F.relu(self.fc1(data.x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc5(x)
        return x

def read_datasets(path):
    data = np.load(path)
    fps = data["fp"].astype(np.float32)
    targets = data["pka"].reshape(-1, 1).astype(np.float32)
    return fps, targets

def numpy_to_tensor(X, y):
    datas = []
    for idx in range(X.shape[0]):
        fp = X[idx].reshape(1, 2048)
        pka = y[idx].reshape(1, 1)
        data = Data(x=torch.tensor(fp, dtype=torch.float32),
                    y=torch.tensor(pka, dtype=torch.float32))
        datas.append(data)
    return datas

def gen_data(X, y):
    data = numpy_to_tensor(X, y)
    train_data, valid_data = train_test_split(data, test_size=0.1)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=128, shuffle=True, drop_last=True)
    return train_loader, valid_loader

def train_step(loader, model, optimizer, device):
    model.train()

    loss_all = 0
    i = 0
    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data.y)
        loss.backward()

        loss_all += loss.item()
        optimizer.step()
        i += 1
    return loss_all / i

def test_step(loader, model,  device):
    model.eval()

    MSE, MAE = 0, 0
    trues, preds = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            pred = output.cpu().numpy()[0][0]
            true = data.y.cpu().numpy()[0][0]

            trues.append(true)
            preds.append(pred)
    MAE = mean_absolute_error(trues, preds)
    MSE = mean_squared_error(trues, preds)
    R2 = r2_score(trues, preds)
    return MAE, MSE, R2


def train(train_loader, test_loader, epochs):
    model = DNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    hist = {"train-loss":[], "test-mae":[], "test-mse":[], "test-r2":[]}
    for epoch in range(epochs):
        weight_path = "models/weight_ap_{}.pth".format(epoch)
        train_loss = train_step(train_loader, model, optimizer, device)
        test_mae, test_mse, test_r2 = test_step(test_loader, model, device)
        hist["train-loss"].append(train_loss)
        hist["test-mae"].append(test_mae)
        hist["test-mse"].append(test_mse)
        hist["test-r2"].append(test_r2)

        if test_mae <= min(hist["test-mae"]):
            torch.save(model.state_dict(), weight_path)

        print(f'Epoch: {epoch}, Train loss: {train_loss:.3}, Test mae: {test_mae:.3}, Test mse: {test_mse:.3}, Test r2: {test_r2:.3}')
    print("---------------------------------\nmin mae: {}\n---------------------------------\n".format(min(hist["test-mae"])))
    return


if __name__=="__main__":
    path_data = "datasets/datasets_ap.npz"
    fps, pkas = read_datasets(path_data)
    train_loader, valid_loader = gen_data(fps, pkas)
    train(train_loader, valid_loader, epochs=50)

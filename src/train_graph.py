import numpy as np
import os.path as osp
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from utils.net import GCNNet


batch_size = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model= GCNNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=10,
                                                       min_lr=0.00001)





def load_data(file_name):
    with open(file_name, "rb") as f:
        conts = f.read()
    data = pickle.loads(conts)
    return data

def prepare_dataset():
    train_data = load_data("train.pickle")
    valid_data = load_data("valid.pickle")
    train_loader = DataLoader(train_data, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    return train_loader, valid_loader

def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data.pka)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def test(loader):
    model.eval()
    correct = 0
    mae = 0
    for data in loader:
        data = data.to(device)
        output = model(data)

        correct += F.mse_loss(output, data.pka).item() * data.num_graphs
        mae += F.l1_loss(output, data.pka).item() * data.num_graphs
    return correct / len(loader.dataset), mae / len(loader.dataset)

train_loader, valid_loader = prepare_dataset()

hist = {"loss":[], "mse":[], "mae":[]}
for epoch in range(1, 1001):
    PATH = "models/weight_{}.pth".format(epoch)
    train_loss = train(epoch)
    mse, mae = test(valid_loader)

    hist["loss"].append(train_loss)
    hist["mse"].append(mse)
    hist["mae"].append(mae)
    if mse <=  min(hist["mse"]):
        torch.save(model.state_dict(), PATH)
    print(f'Epoch: {epoch}, Train loss: {train_loss:.3}, Test_mse: {mse:.3}, Test_mae: {mae:.3}')


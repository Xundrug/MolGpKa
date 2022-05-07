import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import Linear
from torch.nn import BatchNorm1d
from utils.gcn_conv import GCNConv
from torch_geometric.nn import GATConv, GlobalAttention
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader


n_features = 29
hidden = 1024

class GCNNet(torch.nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(n_features, 1024, cached=False) # if you defined cache=True, the shape of batch must be same!
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GCNConv(1024, 512, cached=False)
        self.bn2 = BatchNorm1d(512)
        self.conv3 = GCNConv(512, 256, cached=False)
        self.bn3 = BatchNorm1d(256)
        self.conv4 = GCNConv(256, 512, cached=False)
        self.bn4 = BatchNorm1d(512)
        self.conv5 = GCNConv(512, 1024, cached=False)
        self.bn5 = BatchNorm1d(1024)

        self.att = GlobalAttention(Linear(hidden, 1))
        self.fc2 = Linear(1024, 128)
        self.fc3 = Linear(128, 16)
        self.fc4 = Linear(16, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()

        self.att.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = self.att(x, batch)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(n_features, 1024, heads=1) # if you defined cache=True, the shape of batch must be same!
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GATConv(1024, 512, heads=1)
        self.bn2 = BatchNorm1d(512)
        self.conv3 = GATConv(512, 256, heads=1)
        self.bn3 = BatchNorm1d(256)
        self.conv4 = GATConv(256, 516, heads=1)
        self.bn4 = BatchNorm1d(516)
        self.conv5 = GATConv(516, 1024, heads=1)
        self.bn5 = BatchNorm1d(1024)

        self.fc2 = Linear(1024, 128)
        self.fc3 = Linear(128, 16)
        self.fc4 = Linear(16, 1)

    def forward(self, x, edge_index, batch):

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc4(x)
        return x

dim = 64
class MPNNNet(torch.nn.Module):
    def __init__(self):
        super(MPNNNet, self).__init__()
        self.lin0 = torch.nn.Linear(n_features, dim)

        nn = Sequential(Linear(6, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)

        for i in range(6):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out


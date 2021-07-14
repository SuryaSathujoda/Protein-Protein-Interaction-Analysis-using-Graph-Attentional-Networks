import time
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import PPI
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader

from sklearn.metrics import f1_score

class GAT(nn.Module):
    def __init__(self, input_dim):
        super(GAT, self).__init__()
        self.input_dim = input_dim

        self.conv1 = GATConv(input_dim, 256, 4)
        self.conv2 = GATConv(1024, 256, 4)
        self.conv3 = GATConv(1024, 121, 6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        x = torch.stack(x.split(121, dim=1)).mean(dim=0)

        return x

    def loss(self, preds, label):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(preds, label)

def train(model, loader_tr, loader_val):
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    patience = 20
    epochs_no_improve = 0
    best_f1 = 0
    best_f1_loss = 0
    best_model = model

    for epoch in range(500):
        model.train()
        for batch in loader_tr:
            optimizer.zero_grad()
            pred = model(batch)
            loss = model.loss(pred, batch.y)
            writer.add_scalar('Loss/train', loss, epoch)
            loss.backward()
            optimizer.step()

        running_f1 = 0
        model.eval()
        for batch in loader_val:
            pred = model(batch)
            loss = model.loss(pred, batch.y)
            xs = (pred.detach().numpy() > 0.5)
            ys = batch.y.detach().numpy()
            micro_f1 = f1_score(ys, xs, average='micro')
            running_f1 += micro_f1
            writer.add_scalar('F1/val', micro_f1, epoch)

        running_f1 /= len(loader_val)

        if(epoch % 10 == 0):
            print('Epoch: {}. Micro F1: {:,.4f}'.format(epoch+1, micro_f1))

        if(running_f1 > best_f1):
            best_model = model
            epochs_no_improve = 0
            best_f1 = running_f1
            best_f1_loss = loss
        else:
            epochs_no_improve += 1

        if(epoch > 10 and epochs_no_improve == patience):
            tmp = 'Early Stopping! Epoch: {}, Loss: {}, Micro F1: {}'
            print(tmp.format(epoch, best_f1_loss, best_f1))
            return best_model

    return model

def test(model, data_t):
    running_f1 = 0
    for batch in loader_t:
        model.eval()
        pred = model(batch)
        loss = model.loss(pred, batch.y)
        xs = (pred.detach().numpy() > 0.5)
        ys = batch.y.detach().numpy()
        micro_f1 = f1_score(ys, xs, average='micro')
        running_f1 += micro_f1
        writer.add_scalar('F1/test', micro_f1)
        
    running_f1 /= len(loader_val)
    print('Test Micro F1: {:,.4f}'.format(running_f1))

writer = SummaryWriter()

network = 'PPI'

print('***** '+network+' *****')
data_root = './data/'+network+'/'
dataset_tr = PPI(root=data_root, split='train')
dataset_val = PPI(root=data_root, split='val')
dataset_t = PPI(root=data_root, split='test')
data_tr = dataset_tr[0]

loader_tr = DataLoader(dataset_tr, batch_size=2, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=2, shuffle=False)
loader_t = DataLoader(dataset_t, batch_size=2, shuffle=False)

num_labels = 121
model = GAT(data_tr.num_features)
print('***** '+str(model)+' *****')

tmp_str = 'Num of Nodes: {}, Num of Features: {}, Num of Edges: {}, Num of Labels: {}'
print(tmp_str.format(data_tr.num_nodes, data_tr.num_features, data_tr.num_edges, num_labels))

time_in = time.time()
model = train(model, loader_tr, loader_val)
train_time = time.time() - time_in
print('Train time: {}'.format(train_time))

time_in = time.time()
model = test(model, loader_t)
test_time = time.time() - time_in
print('Test time: {}'.format(test_time))

writer.flush()
writer.close()

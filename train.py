import torch
from torch import nn
from net.ssd import SSD
from preprocessing.data import load_data_xray
from torch import optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 20
batch_size = 32
num_classes = 10
lr = 0.001
momentum = 0.9
weight_decay = 5e-4
net = SSD(num_classes).to(device)
optimizer = optim.SGD(net.parameters(), lr, momentum, weight_decay=weight_decay)

train_iter = load_data_xray(batch_size)

for epoch in range(num_epochs):
    for features, target in train_iter:
        X = features.to(device)
        Y = target.to(device)
        pred = net(X)

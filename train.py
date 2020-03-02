import model
import data_loader
import torch.utils.data.dataloader
import torch.optim as optim
import torch.nn as nn

LEARNING_RATE = 0.001
EPOCH_NUM = 10

net = model.FCN(35)  # 35 classes for Cityscape Dataset
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

for i in range(EPOCH_NUM):

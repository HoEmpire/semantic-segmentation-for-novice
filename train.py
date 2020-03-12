import model
import data_loader
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import time

LEARNING_RATE = 0.0005
EPOCH_NUM = 10
BATCH_SIZE = 8
NUM_WORKERS = 4
USE_GPU = True
USE_PRE_TRAIN = True

if __name__ == '__main__':

    if USE_PRE_TRAIN:
        net = torch.load('model.pkl')
    else:
        net = model.FCN(35)  # 35 classes for Cityscape Dataset

    if USE_GPU:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    raw_data = data_loader.CityScape(all=False)
    transformed_data = []
    composed = transforms.Compose([data_loader.Rescale(256),
                                   data_loader.RandomCrop(224),
                                   data_loader.ToTensor()])
    for d in raw_data:
        transformed_data.append(composed(d))

    dataloaders = DataLoader(transformed_data, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=NUM_WORKERS)

    loss_plt = []
    epoch_plt = []
    plt.ion()
    start_time = time.time()
    for i in range(EPOCH_NUM):
        # forward pass
        running_loss = 0.0

        # iterate the data
        for batches in dataloaders:
            if USE_GPU:
                inputs = batches['image'].cuda()
                labels = batches['label'].squeeze().cuda()
            else:
                inputs = batches['image']
                labels = batches['label'].squeeze()

            # zero the gradient
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # get the loss of each epoch
            running_loss += loss.item() * inputs.size(0)

        # visualize the process
        loss_plt.append(running_loss)
        epoch_plt.append(i)
        plt.cla()
        plt.plot(epoch_plt, loss_plt, 'r-', lw=5)
        plt.text(i/2, loss_plt[0], 'Loss=%.4f' % running_loss,
                 fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
        print('Loss: {:.4f}'.format(running_loss))

    end_time = time.time()

    plt.ioff()
    plt.show()
    torch.save(net, 'model.pkl')
    print('Training time: {:.4f}'.format(end_time-start_time))

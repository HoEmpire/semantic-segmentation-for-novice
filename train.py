import model
import data_loader
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

LEARNING_RATE = 0.001
EPOCH_NUM = 10
BATCH_SIZE = 16
NUM_WORKERS = 4
USE_GPU = True

if __name__ == '__main__':
    if USE_GPU:
        net = model.FCN(35).cuda()  # 35 classes for Cityscape Dataset
    else:
        net = model.FCN(35)

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

    for i in range(EPOCH_NUM):
        # forward
        running_loss = 0.0
        # 迭代数据.
        for batches in dataloaders:
            if USE_GPU:
                inputs = batches['image'].cuda()
                labels = batches['label'].squeeze().cuda()
            else:
                inputs = batches['image']
                labels = batches['label'].squeeze()

            # 零参数梯度
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss
        print('Loss: {:.4f}'.format(epoch_loss))

    net.save('model.pkl')

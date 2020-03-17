import torch.nn as nn
import torchvision
import torch.functional as F
import torch
import numpy as np
import time

BIAS_INIT = 0.01


def bilinear_kernel_init(conv_layer):
    '''
    return a conv_layer with initialized weight and bias 
    '''
    kernel_size = conv_layer.kernel_size
    in_channels = conv_layer.in_channels
    out_channels = conv_layer.out_channels

    factor = (kernel_size[0] + 1) // 2
    if kernel_size[0] % 2 == 1:
        center_row = factor - 1
    else:
        center_row = factor - 0.5

    factor = (kernel_size[1] + 1) // 2
    if kernel_size[1] % 2 == 1:
        center_col = factor - 1
    else:
        center_col = factor - 0.5

    og = np.ogrid[:kernel_size[0], :kernel_size[1]]
    filt = (1 - abs(og[0] - center_row) / factor) * \
        (1 - abs(og[1] - center_col) / factor)
    weight = np.zeros((out_channels, in_channels, kernel_size[0],
                       kernel_size[1]), dtype='float32')
    for i in range(out_channels):
        for j in range(in_channels):
            weight[i][j] = filt

    conv_layer.weight.data = torch.from_numpy(weight)
    conv_layer.bias.data = BIAS_INIT * torch.ones(out_channels)
    return conv_layer


class FCN(nn.Module):
    def __init__(self, class_num):
        super(FCN, self).__init__()
        resNet = torchvision.models.resnet34(pretrained=True)
        # resize the image from 224*448 to 224*224
        self.conv0 = nn.Conv2d(3, 3, 5, stride=(1, 2), padding=(2, 2))
        self.conv0 = bilinear_kernel_init(self.conv0)

        self.layer123 = nn.Sequential(*list(resNet.children())[:-4])
        self.layer4 = nn.Sequential(*list(resNet.children())[-4])
        self.layer5 = nn.Sequential(*list(resNet.children())[-3])

        # resize the number of channels
        self.conv1 = nn.Conv2d(128, class_num, 1)
        self.conv2 = nn.Conv2d(256, class_num, 1)
        self.conv3 = nn.Conv2d(512, class_num, 1)

        self.upsample_2x = nn.ConvTranspose2d(class_num, class_num, 4, 2, 1)
        self.upsample_4x = nn.ConvTranspose2d(class_num, class_num, 4, 2, 1)
        self.upsample_8x = nn.ConvTranspose2d(class_num, class_num, 8, 8, 0)
        self.resize = nn.ConvTranspose2d(
            class_num, class_num, 5, (1, 2), (2, 2), (0, 1))

        self.upsample_2x = bilinear_kernel_init(self.upsample_2x)
        self.upsample_4x = bilinear_kernel_init(self.upsample_4x)
        self.upsample_8x = bilinear_kernel_init(self.upsample_8x)
        self.resize = bilinear_kernel_init(self.resize)

    def forward(self, x):
        x = self.conv0(x)
        s1 = self.layer123(x)
        s2 = self.layer4(s1)
        s3 = self.layer5(s2)

        s1 = self.conv1(s1)
        s2 = self.conv2(s2)
        s3 = self.conv3(s3)

        x = self.upsample_2x(s3) + s2
        x = self.upsample_4x(x) + s1
        x = self.upsample_8x(x)
        x = self.resize(x)

        return x


'''
dummpy test
'''
if __name__ == '__main__':
    model = FCN(20)
    input = torch.rand(1, 3, 224, 1024)
    start = time.time()
    output = model(input)
    end = time.time()
    print('forward pass time:{:.4f}', end-start)
    # print(ouput)
    # print(ouput.shape)
    target = torch.zeros(1, 224, 1024, dtype=int)
    _, preds = torch.max(output, 1)

    loss_fun = nn.CrossEntropyLoss()
    print(output.shape)
    print(target.shape)
    loss = loss_fun(output, target)
    print(loss)

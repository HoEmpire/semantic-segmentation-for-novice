import torch.nn as nn
import torchvision
import torch.functional as F
import torch
import numpy as np

BIAS_INIT = 0.01


def bilinear_kernel_init(conv_layer):
    '''
    return a conv_layer with initialized weight and bias 
    '''
    kernel_size = conv_layer.kernel_size[0]
    in_channels = conv_layer.in_channels
    out_channels = conv_layer.out_channels

    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)
    weight = np.zeros((out_channels, in_channels, kernel_size,
                       kernel_size), dtype='float32')
    for i in range(out_channels):
        for j in range(in_channels):
            weight[i][j] = filt

    conv_layer.weight.data = torch.from_numpy(weight)
    conv_layer.bias.data = BIAS_INIT * torch.ones(out_channels)
    return conv_layer


class FCN(nn.Module):
    def __init__(self, class_num):
        super(FCN, self).__init__()
        resNet = torchvision.models.resnet101(pretrained=True)
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

        self.upsample_2x = bilinear_kernel_init(self.upsample_2x)
        self.upsample_4x = bilinear_kernel_init(self.upsample_4x)
        self.upsample_8x = bilinear_kernel_init(self.upsample_8x)

    def forward(self, x):
        s1 = self.layer123(x)
        s2 = self.layer4(s1)
        s3 = self.layer5(s2)

        s1 = self.conv1(s1)
        s2 = self.conv2(s2)
        s3 = self.conv3(s3)

        x = self.upsample_2x(s3) + s2
        x = self.upsample_4x(x) + s1
        x = self.upsample_8x(x)

        return x


'''
dummpy test
'''
if __name__ == '__main__':
    model = FCN(20)
    input = torch.rand(10, 3, 224, 224)
    output = model(input)
    # print(ouput)
    # print(ouput.shape)
    target = torch.zeros(10, 224, 224, dtype=int)
    _, preds = torch.max(output, 1)

    loss_fun = nn.CrossEntropyLoss()
    # print(out)
    # print(target)
    loss = loss_fun(output, target)
    print(loss)

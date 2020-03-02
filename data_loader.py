import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

# PATH of your dataset
PATH_X = './data/leftImg8bit_trainvaltest/leftImg8bit'
PATH_Y = './data/gtFine_trainvaltest/gtFine'


class Rescale(object):
    """
    将样本中的图片和label重新缩放到给定大小
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        T = transforms.Resize(self.output_size)
        image = T(image)
        label = T(label)

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    随机裁剪样本中的图像.    
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        w, h = image.size
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image.crop((left, top, left + new_w, top + new_h))
        label = label.crop((left, top, left + new_w, top + new_h))

        return {'image': image, 'label': label}


class ToTensor(object):
    """
    将样本中的ndarrays转换为Tensors.
    """

    def __call__(self, sample):
        return {'image': transforms.ToTensor()(sample['image']),
                'label': (transforms.ToTensor()(sample['label'])*255).long()}


class CityScape(object):

    def __init__(self, train=True, all=True):

        if train:
            root_x = os.path.join(PATH_X, 'train')
            root_y = os.path.join(PATH_Y, 'train')
        else:
            root_x = os.path.join(PATH_X, 'test')
            root_y = os.path.join(PATH_Y, 'test')

        city = os.listdir(root_x)
        root_file_x = []
        root_file_y = []

        if all:
            for c in city:
                root_file_x.append(os.path.join(root_x, c))
                root_file_y.append(os.path.join(root_y, c))
        else:
            root_file_x.append(os.path.join(root_x, city[0]))
            root_file_y.append(os.path.join(root_y, city[0]))

        self.x = []
        self.y = []

        for r_x, r_y in zip(root_file_x, root_file_y):

            for f in os.listdir(r_x):
                self.x.append(Image.open(os.path.join(r_x, f)))

            all_file_name_y = os.listdir(r_y)
            file_name_y = []
            for f in all_file_name_y:
                if f[-8] == 'l':
                    file_name_y.append(f)

            for f in file_name_y:
                self.y.append(Image.open(os.path.join(r_y, f)))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, id):
        target = {}
        target['image'] = self.x[id]
        target['label'] = self.y[id]
        return target


'''
dummy test
'''
if __name__ == '__main__':
    a = CityScape(train=True, all=False)
    print(len(a))
    print(type(a[0]))
    print(a[0])
    composed = transforms.Compose([Rescale(256),
                                   RandomCrop(224),
                                   ToTensor()])
    b = []
    for aa in a:
        b.append(composed(aa))
        aa = b[-1]

    print(type(b[0]['label']))
    print(type(a[0]['label']))

    print(np.unique(b[0]['label'].numpy()))

import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import random

# PATH of your dataset
PATH_X = './data/leftImg8bit_trainvaltest/leftImg8bit'
PATH_Y = './data/gtFine_trainvaltest/gtFine'


class Rescale(object):
    """
    rescale the size of the image
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        T = transforms.Resize(self.output_size)
        image = T(image)
        label = T(label)

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    crop the image randomly
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
    change type from ndarray to tensor
    """

    def __call__(self, sample):
        return {'image': transforms.ToTensor()(sample['image']),
                'label': (transforms.ToTensor()(sample['label'])*255).long().squeeze()}
        # change range from [0,1] to [0, 255]


class CityScape(object):
    """
            arg:
            train: load training data or not, if not load validation data
            rand: define the proportion of the loaded data, range from 0~1, default
                  is -1 as loading all the data
    """

    def __init__(self, train=True, rand=-1):

        if train:
            root_x = os.path.join(PATH_X, 'train')
            root_y = os.path.join(PATH_Y, 'train')
        else:
            root_x = os.path.join(PATH_X, 'val')
            root_y = os.path.join(PATH_Y, 'val')

        city = os.listdir(root_x)
        city_file_x = []
        city_file_y = []

        for c in city:
            file_name_x = os.listdir(os.path.join(root_x, c))
            all_file_y = os.listdir(os.path.join(root_y, c))
            file_name_y = []

            for i in range(len(file_name_x)):
                file_name_x[i] = os.path.join(root_x, c, file_name_x[i])

            for a in all_file_y:
                if a.endswith('_labelIds.png'):
                    file_name_y.append(os.path.join(root_y, c, a))

            city_file_x.append(file_name_x)
            city_file_y.append(file_name_y)

        root_file_x = []
        root_file_y = []

        if rand == -1:
            for i in range(len(city)):
                for (x, y) in zip(city_file_x[i], city_file_y[i]):
                    root_file_x.append(x)
                    root_file_y.append(y)
        else:
            for i in range(len(city)):
                for (x, y) in zip(city_file_x[i], city_file_y[i]):
                    if(random.random() < rand):
                        root_file_x.append(x)
                        root_file_y.append(y)

        if train:
            composed = transforms.Compose([Rescale(256),
                                           RandomCrop((224, 448)),
                                           ToTensor()])
        else:
            composed = transforms.Compose([Rescale((224, 448)),
                                           ToTensor()])
        self.data = []
        sample = {}
        for r_x, r_y in zip(root_file_x, root_file_y):
            sample['image'] = Image.open(r_x)
            sample['label'] = Image.open(r_y)
            self.data.append(composed(sample))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        return self.data[id]


'''
dummy test
'''
if __name__ == '__main__':
    a = CityScape(train=True, rand=0.005)
    print(len(a))
    print(type(a[0]))
    print(type(a[0]['label']))
    print(a[0]['label'].shape)

    print(np.unique(a[0]['label'].numpy()))
    plt.figure(0)
    plt.imshow(a[0]['image'].numpy().transpose(1, 2, 0))
    plt.figure(1)
    plt.imshow(a[0]['label'].numpy())
    plt.show()

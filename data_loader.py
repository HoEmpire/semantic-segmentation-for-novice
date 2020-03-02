import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

# PATH of your dataset
PATH_X = './data/leftImg8bit_trainvaltest/leftImg8bit'
PATH_Y = './data/gtFine_trainvaltest/gtFine'


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
                self.x.append(np.array(Image.open(os.path.join(r_x, f)))/255)

            all_file_name_y = os.listdir(r_y)
            file_name_y = []
            for f in all_file_name_y:
                if f[-8] == 'l':
                    file_name_y.append(f)

            for f in file_name_y:
                self.y.append(np.array(Image.open(os.path.join(r_y, f))))

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
    print(a[0]['label'].shape)
    plt.imshow(a[10]['label'])
    plt.show()

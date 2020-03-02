import torch
import data_loader

raw_data = data_loader.CityScape(train=True, all=False)
transformed_data = []
composed = transforms.Compose([data_loader.Rescale(256),
                               data_loader.RandomCrop(224),
                               data_loader.ToTensor()])

import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Train(Dataset):
    def __init__(self, direc, transforms=transforms.Compose([transforms.ToTensor()])):
        super(Train, self).__init__()
        self.transforms = transforms
        self.paths = []
        self.annotations = []

        with open(direc + 'train.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.split(' ')
                filename = content[0]
                x1 = round(float(content[1]))
                y1 = round(float(content[2]))
                x2 = round(float(content[3]))
                y2 = round(float(content[4]))
                self.paths.append(direc + filename)
                self.annotations.append([x1, y1, x2, y2])

    def __getitem__(self, index):
        path = self.paths[index]
        item = Image.open(path).convert('RGB')
        item.resize((64, 64))
        if self.transforms is not None:
            item = self.transforms(item)
            annotation = torch.Tensor(self.annotations[index])
        return item, annotation

    def __len__(self):
        return len(self.paths)


class Test(Dataset):
    def __init__(self, direc, transforms=transforms.Compose([transforms.ToTensor()])):
        super(Test, self).__init__()
        self.transforms = transforms
        self.paths = []
        self.annotations = []

        with open(direc + 'test.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.split(' ')
                filename = content[0]
                x1 = round(float(content[1]))
                y1 = round(float(content[2]))
                x2 = round(float(content[3]))
                y2 = round(float(content[4]))
                self.paths.append(direc + filename)
                self.annotations.append([x1, y1, x2, y2])

    def __getitem__(self, index):
        path = self.paths[index]
        item = Image.open(path).convert('RGB')
        if self.transforms is not None:
            item = self.transforms(item)

        return item, self.annotations[index]

    def __len__(self):
        return len(self.paths)
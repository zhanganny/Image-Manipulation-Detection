import os
import yaml
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


from data import Train, Test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--iteration', default=2)
    args = parser.parse_args()

    trainset = Train('D:/Datasets/coco_tampered/')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    model = None # TODO: create faster R-CNN model
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(5e-3, 5e-3))
    lossRPN = torch.nn.MSELoss().cuda()
    lossTemper = torch.nn.MSELoss().cuda()
    lossBBox = torch.nn.square().cuda()

    for epoch in range(args.iteration):
        # model.train()
        for step, (imgs, annotations) in enumerate(trainloader):
            # print(imgs, annotations)

            # model.forword

            # calculate the loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

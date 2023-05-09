import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from data import Train, Test
from model import Fusion_FasterRCNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--iteration', default=20)
    args = parser.parse_args()

    trainset = Train('D:/Datasets/coco_tampered/')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    model = Fusion_FasterRCNN(512).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(5e-3, 5e-3))

    for epoch in range(args.iteration):
        model.train()
        for step, (imgs, annotations) in enumerate(trainloader):
            print(imgs.size(), annotations)
            imgs = imgs.cuda()
            annotations = annotations.cuda()

            model(imgs, annotations=annotations)
            optimizer.zero_grad()
            loss = model.loss()
            model.zero_loss()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            print("checkpoint {}".format(epoch))
            model.eval()

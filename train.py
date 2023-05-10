import os
import cv2
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from data import Train, Test
from model import Fusion_FasterRCNN


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='D:/Datasets/coco_tampered/')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--iteration', default=5)
    args = parser.parse_args()

    use_cuda = True

    trainset = Train(args.dataset)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    model = Fusion_FasterRCNN()

    if use_cuda:
        model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(5e-3, 5e-3))

    for epoch in range(args.iteration):
        model.train()
        for step, (imgs, annotations) in enumerate(trainloader):
            # print(imgs.size(), annotations)
            if use_cuda:
                imgs = imgs.cuda()
                annotations = annotations.cuda()
            try:
                model(imgs, annotations=annotations)
                optimizer.zero_grad()
                loss = model.loss()
                model.zero_loss()
                loss.backward()
                optimizer.step()
                print("epoch: {}\tstep: {}\tloss: {}".format(epoch, step, loss))
            except:
                print("epoch: {}\tstep: {}\tskip".format(epoch, step))

        if epoch % 1 == 0:
            print("checkpoint {}".format(epoch))
            model.eval()
            with open(args.dataset + 'test.txt', 'r') as f:
                lines = f.readlines()
                line = lines[0]
                content = line.split(' ')
                filename = content[0]
                x1 = round(float(content[1]))
                y1 = round(float(content[2]))
                x2 = round(float(content[3]))
                y2 = round(float(content[4]))
            transform = transforms.Compose([transforms.ToTensor()])
            test_img = Image.open(args.dataset + filename).convert('RGB')
            test_img = test_img.resize((int(test_img.size[0] / 2), 
                                        int(test_img.size[1] / 2)))
            test_img = transform(test_img)
            test_annotation = torch.Tensor([x1, y1, x2, y2]) / 2
            if use_cuda:
                test_img = test_img.cuda()
                test_annotation = test_annotation.cuda()
            roi_cls_locs, roi_scores = model(test_img.unsqueeze(0), annotations=test_annotation.unsqueeze(0))
            test_img = cv2.imread(args.dataset + filename)
            # test_img = cv2.rectangle(test_img, (x1, y1), (x2, y2), (255, 0, 255))
            # test_img = cv2.putText(test_img, str(roi_cls_locs.size(0)), (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            for i in range(roi_cls_locs.size(0)):
                if roi_scores[i, 0] > 0.5:
                    continue
                test_img = cv2.rectangle(test_img, 
                                        (int(2 * roi_cls_locs[i][0]), int(2 * roi_cls_locs[i][1])), 
                                        (int(2 * roi_cls_locs[i][2]), int(2 * roi_cls_locs[i][3])),
                                        (255, 0, 255))
            cv2.imwrite('./result/epoch{}.jpg'.format(epoch), test_img)
            torch.save(model.state_dict(), './checkpoints/epoch{}.pt'.format(epoch))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class SRMLayer(nn.Module):
    def __init__(self):
        super(SRMLayer, self).__init__()
        self.q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        filter1 = np.array(filter1) / 4
        filter2 = np.array(filter2) / 12
        filter3 = np.array(filter3) / 2
        filters = np.array([[filter1, filter2, filter3]])  # shape=(5,5,3)
        filters = torch.tensor(filters, dtype=torch.float)
        # print(filters)

        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2,
                              bias=False)  # 输入和输出通道数都为3，卷积核大小为5x5，stride为1，padding为2（same填充方式填充kernel_size/2（向下取整）；valid不填充）
        self.conv.weight = nn.Parameter(filters)

    def forward(self, x):
        # x = F.conv2d(input=x, weight=self.conv.weight, bias=None,
        #              stride=1, padding=2, dilation=1, groups=1)
        x = self.conv(x)
        return x


def tst_srm(img_path, noise_path):
    # 读取图片
    # img = Image.open(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 定义图像预处理的转换
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # 将图像转为张量(batch_size, channels, height, width)
    ])

    # 对图片进行预处理转换
    input_tensor = preprocess(img)
    # print(input_tensor.size())

    # 添加批次维度
    input_tensor = input_tensor.unsqueeze_(0)
    # print(input_tensor.size())

    # 创建SRMLayer实例
    srm_layer = SRMLayer()

    # 将图片输入SRMLayer进行处理
    with torch.no_grad():
        noise_img = srm_layer(input_tensor)
    # noise_img = srm_layer(input_tensor)

    # 将张量转换为图片
    noise_img = noise_img.squeeze(0).permute(1, 2, 0).numpy()
    noise_img = cv2.cvtColor(noise_img, cv2.COLOR_RGB2BGR)
    noise_img = (noise_img * 255).astype('uint8')

    # 显示噪声图片
    cv2.imshow('Noise Image', noise_img)
    cv2.imwrite(noise_path, noise_img)


if __name__ == '__main__':
    img_path = '../../lena.png'
    noise_path = '../../noise.png'
    tst_srm(img_path, noise_path)

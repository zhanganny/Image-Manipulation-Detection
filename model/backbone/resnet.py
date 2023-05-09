import cv2
import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url

import torchvision
import torchvision.transforms as transforms


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        # n, h, w, inplanes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, 
                               stride=stride, dilation=dilation, 
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # n, h, w, planes
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                               stride=stride, dilation=dilation, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # n, h, w, planes
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, 
                               stride=stride, dilation=dilation,
                               padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        # n, h, w, 4 * planes
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride
        self.dialation=dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64

        # h, w, 3 -> h/2, w/2, 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, 
                               stride=2, dilation=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # h/2, w/2, 64 -> h/4, w/4, 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    padding=0, ceil_mode=True)

        # h/2, w/2, 64 -> h/4, w/4, 256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # h/2, w/2, 64 -> h/8, w/8, 512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # h/2, w/2, 64 -> h/16, w/16, 1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'], 
                                            model_dir="./model_data")
        # state_dict = model_zoo.load_url(model_urls['resnet50'], model_dir="./model_data")
        model.load_state_dict(state_dict, strict=True)

    # 获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    features = list([model.conv1, 
                     model.bn1, 
                     model.relu, 
                     model.maxpool, 
                     model.layer1, 
                     model.layer2, 
                     model.layer3])
    features = nn.Sequential(*features)

    # 获取分类部分，从model.layer4到model.avgpool
    classifier  = list([model.layer4, 
                        model.avgpool])
    classifier  = nn.Sequential(*classifier)

    return features, classifier


def resnet101(pretrained=True):
    assert pretrained == True
    resnet_net = torchvision.models.resnet101(pretrained=True)
    modules = list(resnet_net.children())

    # print(modules[7:])

    encoder = nn.Sequential(*modules[:7])
    decoder = nn.Sequential(*modules[7:-1])

    return encoder, decoder
    

if __name__ == "__main__":
    img = cv2.imread("lena.png")
    img = cv2.resize(img, (4096, 4096))
    transformer = transforms.ToTensor()
    img = transformer(img)
    imgs = img.unsqueeze(0)

    encoder, decoder = resnet101()
    
    print(imgs.size())      # [1, 3, 512, 512]
    feature = encoder(imgs)
    print(feature.size())   # [1, 1024, 32, 32]
    cls = decoder(feature)
    print(cls.size())       # [1, 2048, 1, 1]

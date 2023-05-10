import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import RoIPool
from utils import bbox_match
from CompactBilinearPooling import CompactBilinearPooling


warnings.filterwarnings("ignore")


class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier, mode='training'):
        super(Resnet50RoIHead, self).__init__()
        self.mode = mode

        self.roiPool = RoIPool((roi_size, roi_size), spatial_scale)

        self.classifier = classifier
        self.bbox_pred = nn.Linear(2048, 4 * n_class)

        # self.bilinear = nn.Bilinear(1024, 1024, 8)
        self.bilinear = CompactBilinearPooling(1024, 1024, 8, cuda=False)
        self.cls_pred = nn.Linear(8, n_class + 1)

        normal_init(self.bbox_pred, 0, 0.001)
        normal_init(self.cls_pred, 0, 0.01)
        
        self.smoothL1Loss = nn.SmoothL1Loss()
        self.crossEntropyLoss = nn.CrossEntropyLoss()
        self.loss_tamper = 0
        self.loss_bbox = 0

    def forward(self, x, x_noise, rois, roi_indices, img_size, annotations=None):
        """
            x: RGB features
            x_noise: Noise featuers
            rois: Region of interest
            roi_indices: Source image index in batch
            img_size: Source image size
            annotations: [[x1, y1, x2, y2], ...] of the batch
        """
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()

        # step 1: RoI Pooling
        #   - Two streams share the same feature map
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]
        indices_and_rois = torch.cat([roi_indices, rois_feature_map], dim=1)
        #   - RGB and Noise RoI feature -> [N, C, roi_size, roi_size]
        pool_rgb = self.roiPool(x, indices_and_rois)
        pool_noise = self.roiPool(x_noise, indices_and_rois)

        # step 2: BBox Pred (RGB channels only)
        #   - 利用classifier网络进行特征提取
        fc7_rgb = self.classifier(pool_rgb)
        #   - 当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        fc7_rgb = fc7_rgb.view(fc7_rgb.size(0), -1)
        #   - fc7即为roi feature
        roi_bbox = self.bbox_pred(fc7_rgb)
        # roi_bbox = roi_bbox.view(n, -1, roi_bbox.size(1))

        # step 3: Bilnear Pooling
        bi_feature = self.bilinear(pool_rgb, pool_noise)

        # step 4: Class Pres        
        # fc7_bilinear = self.classifier(bi_feature)
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        # fc7_bilinear = fc7_rgb.view(fc7_binear.size(0), -1)
        # fc7即为roi feature
        roi_scores = self.cls_pred(bi_feature)
        # roi_scores = roi_scores.view(n, -1, roi_scores.size(1))

        # 训练时计算 RPN Loss
        if self.mode == 'training': 
            assert annotations is not None
            # 每个roi的标签，有效roi下标，正样本roi下标
            labels, valid_indices, pos_indices = \
                bbox_match(roi_bbox, annotations, neg_thres=0.3, pos_thres=0.7)

            self.loss_tamper = self.smoothL1Loss(roi_bbox[pos_indices], annotations[0])
            self.loss_bbox = self.crossEntropyLoss(roi_scores[valid_indices, 1].unsqueeze(1), labels[valid_indices])

        return roi_bbox, roi_scores

    def train(self, mode=True):
        super().train()
        self.classifier.eval()

    def zero_loss(self):
        self.loss_tamper = 0
        self.loss_bbox = 0

    def loss(self):
        return self.loss_tamper + self.loss_bbox

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

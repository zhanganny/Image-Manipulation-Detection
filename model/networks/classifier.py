import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import RoIPool
from utils.utils_bbox import bbox_match
from CompactBilinearPooling import CompactBilinearPooling

warnings.filterwarnings("ignore")


class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier, mode='training'):
        super(Resnet50RoIHead, self).__init__()
        self.mode = mode
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        #--------------------------------------#
        self.bbox_pred = nn.Linear(2048, 4)
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.cls_pred = nn.Linear(2048, 2)
        #----------------------------，-------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.bbox_pred, 0, 0.001)
        normal_init(self.cls_pred, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)
        # self.bilinear = nn.Bilinear(1024, 1024, 16384)
        self.bilinear = CompactBilinearPooling(1024, 1024, 2048, cuda=False)

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
        # rois        = torch.flatten(rois, 0, 1)
        # roi_indices = torch.flatten(roi_indices, 0, 1)
        
        rois_feature_map_rgb = torch.zeros_like(rois)
        rois_feature_map_noise = torch.zeros_like(rois)

        # step 1: RoI Pooling
        #   - RGB RoI feature
        rois_feature_map_rgb[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map_rgb[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]
        indices_and_rois_rgb = torch.cat([roi_indices[:, None], rois_feature_map_rgb], dim=1)
        pool_rgb = self.roi(x, indices_and_rois_rgb)
        #   - Noise RoI feature
        rois_feature_map_noise[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x_noise.size()[3]
        rois_feature_map_noise[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x_noise.size()[2]
        indices_and_rois_noise = torch.cat([roi_indices[:, None], rois_feature_map_noise], dim=1)
        pool_noise = self.roi(x_noise, indices_and_rois_noise)

        # step 2: BBox Pred (RGB channels only)
        #   - 利用classifier网络进行特征提取
        fc7_rgb = self.classifier(pool_rgb)
        #   - 当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        fc7_rgb = fc7_rgb.view(fc7_rgb.size(0), -1)
        #   - fc7即为roi feature
        roi_bbox = self.bbox_pred(fc7_rgb)
        roi_bbox = roi_bbox.view(n, -1, roi_bbox.size(1))

        # step 3: Bilnear Pooling
        bi_feature = self.bilinear(pool_rgb, pool_noise)

        # step 4: Class Pres        
        # fc7_bilinear = self.classifier(bi_feature)
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        # fc7_bilinear = fc7_rgb.view(fc7_binear.size(0), -1)
        # fc7即为roi feature
        roi_scores = self.cls_pred(fc7_binear)
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))

        if self.mode == 'training':
            """
            所有 roi 中: 
                对于 IoU >= 0.7 的，认为其 p* 是 1 
                对于 IoU < 0.3 的，认为其 p* 是 0
                对于 0.3 <= IoU < 0.7 的，不计算 Loss
            """
            assert annotations is not None
            gt_bboxes = []
            for i in len(roi):
                roi_index = roi_indices[i]
                annotation = annotations[roi_index]
                gt_bboxes.append(annotation)

            # 每个roi的标签，有效roi下标，正样本roi下标
            labels, valid_indices, pos_indices = \
                bbox_match(roi_bbox, gt_bboxes, neg_thres=0.3, pos_thres=0.7)

            self.loss_tamper += nn.CrossEntropyLoss(roi_scores, labels[valid_indices])
            self.loss_bbox += nn.SmoothL1Loss(roi_bbox, gt_bboxes[pos_indices])

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

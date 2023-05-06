import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool

warnings.filterwarnings("ignore")


class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        #--------------------------------------#
        self.cls_loc = nn.Linear(2048, n_class * 4)
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.score = nn.Linear(2048, n_class)
        #-----------------------------------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, x_noise, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)
        
        rois_feature_map_rgb = torch.zeros_like(rois)
        rois_feature_map_noise = torch.zeros_like(rois)

        # rbg
        rois_feature_map_rgb[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map_rgb[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]
        indices_and_rois_rgb = torch.cat([roi_indices[:, None], rois_feature_map_rgb], dim=1)
        #-----------------------------------#
        #   利用建议框对公用特征层进行截取
        #-----------------------------------#
        pool_rgb = self.roi(x, indices_and_rois_rgb)
        #-----------------------------------#
        #   利用classifier网络进行特征提取
        #-----------------------------------#
        fc7_rgb = self.classifier(pool_rgb)
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        #--------------------------------------------------------------#
        fc7_rgb = fc7_rgb.view(fc7_rgb.size(0), -1)
        # fc7即为roi feature
        roi_cls_locs_rgb    = self.cls_loc(fc7_rgb)
        roi_cls_locs_rgb    = roi_cls_locs_rgb.view(n, -1, roi_cls_locs.size(1))
        
        # noise
        rois_feature_map_noise[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x_noise.size()[3]
        rois_feature_map_noise[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x_noise.size()[2]
        indices_and_rois_noise = torch.cat([roi_indices[:, None], rois_feature_map_noise], dim=1)
        pool_noise = self.roi(x_noise, indices_and_rois_noise)
        fc7_noise = self.classifier(pool_noise)
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        #--------------------------------------------------------------#
        fc7_noise = fc7_rgb.view(fc7_noise.size(0), -1)
        # fc7即为roi feature
        roi_cls_locs_noise    = self.cls_loc(fc7_noise)
        roi_cls_locs_noise    = roi_cls_locs_noise.view(n, -1, roi_cls_locs.size(1))


        # For bounding box regression, we only use the RGB channels
        roi_scores      = self.score(fc7_rgb)
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs_rgb, roi_cls_locs_noise, roi_scores

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

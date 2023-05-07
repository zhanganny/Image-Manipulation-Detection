import torch.nn as nn

from .backbone import make_resnet50
from .networks import Resnet50RoIHead  #, VGG16RoIHead
from .networks import RPN
# from model.networs.vgg16 import decom_vgg16


class Fusion_FasterRCNN(nn.Module):
    def __init__(self, 
                 num_classes,  
                 mode = "training",
                 feat_stride = 16,
                 anchor_scales = [8, 16, 32],
                 anchor_ratios = [0.5, 1, 2],
                 backbone = 'resnet50',
                 pretrained = False
                ):
        super(Fusion_FasterRCNN, self).__init__()
        self.mode = "tarining"
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios

        self.srm_filter_layer = None #TODO
        
        if backbone == 'resnet50':
            self.extractor, classifier = make_resnet50(pretrained)
        elif backbone == 'resnet101':
            raise NotImplementedError()

        self.rpn = RPN(
                in_channels=1024, 
                mid_channels=512,
                anchor_ratios=self.anchor_ratios,
                anchor_scales=self.anchor_scales,
                feat_stride=self.feat_stride,
                mode=self.mode
            )

        self.head = Resnet50RoIHead(
                n_class=num_classes + 1,
                roi_size=14,
                spatial_scale=1,
                classifier=classifier
            )
            
    def forward(self, x, scale=1., mode="forward", annotations=None):
        if mode == "forward":
            #---------------------------------#
            #   计算输入图片的大小
            #---------------------------------#
            img_size        = x.shape[2:]
            #---------------------------------#
            #   利用主干网络提取特征
            #   RGB 和 noise使用同一backbone
            #---------------------------------#
            base_feature_rgb    = self.extractor.forward(x)
            noise_x = self.srm_filter_layer(x)
            base_feature_rgb    = self.extractor.forward(x)
            base_feature_noise = self.extractor.forward(noise_x)
            #---------------------------------#
            #   获得建议框
            #   The RGB and noise streams share the same region proposals 
            #   from RPN network which only uses RGB features as input.
            #---------------------------------#
            _, _, rois, roi_indices, _  = self.rpn.forward(base_feature_rgb, img_size, scale, annotations)
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #---------------------------------------#
            roi_cls_locs, roi_scores    = self.head.forward(base_feature_rgb, base_feature_noise, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            #---------------------------------#
            #   利用主干网络提取特征
            #---------------------------------#
            base_feature    = self.extractor.forward(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            #---------------------------------#
            #   获得建议框
            #---------------------------------#
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #---------------------------------------#
            roi_cls_locs, roi_scores    = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def zero_loss(self):
        self.rpn.zero_loss()
        self.head.zero_loss()

    def loss(self):
        return self.rpn.loss() + self.head.loss()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
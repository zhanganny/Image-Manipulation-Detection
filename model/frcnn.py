import torch.nn as nn
import torchvision

from model.backbone import resnet50, resnet101
from model.networks import Resnet50RoIHead
from model.networks import RPN, SRMLayer


class Fusion_FasterRCNN(nn.Module):
    def __init__(self, 
                 num_classes=1,  
                 mode = "training",
                 feat_stride = 16,
                 anchor_scales = [8, 16, 32],
                 anchor_ratios = [0.5, 1, 2],
                 backbone = 'resnet50',
                 pretrained = True
                ):
        super(Fusion_FasterRCNN, self).__init__()
        self.mode = mode
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios

        self.srm_filter_layer = SRMLayer()
        self.extractor, self.classifier = eval(backbone)(pretrained)

        self.rpn = RPN(
                in_channels=1024, 
                mid_channels=512,
                anchor_ratios=self.anchor_ratios,
                anchor_scales=self.anchor_scales,
                feat_stride=self.feat_stride,
                mode=self.mode
            )

        self.head = Resnet50RoIHead(
                n_class=num_classes,
                roi_size=7,
                spatial_scale=0.0625, # 1 / 16
                classifier=self.classifier
            )
            
    def forward(self, x, scale=1., mode="forward", annotations=None):
        if mode == "forward":
            # 计算输入图片的大小 [H, W]
            img_size = x.shape[2:]
            noise_x = self.srm_filter_layer(x)
            # Stage 1
            base_feature_rgb = self.extractor(x)
            base_feature_noise = self.extractor(noise_x)
            _, _, rois, roi_indices, _  = self.rpn(base_feature_rgb, img_size, scale, annotations)
            # Stage 2
            roi_cls_locs, roi_scores = self.head(x=base_feature_rgb, 
                                                 x_noise=base_feature_noise, 
                                                 rois=rois, 
                                                 roi_indices=roi_indices, 
                                                 img_size=img_size, 
                                                 annotations=annotations)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            #---------------------------------#
            #   利用主干网络提取特征
            #---------------------------------#
            base_feature = self.extractor(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            #---------------------------------#
            #   获得建议框
            #---------------------------------#
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #---------------------------------------#
            roi_cls_locs, roi_scores    = self.head(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def train(self, mode=True):
        super().train()
        self.extractor.eval()
        self.classifier.eval()
        self.srm_filter_layer.eval()

    def zero_loss(self):
        self.rpn.zero_loss()
        self.head.zero_loss()

    def loss(self):
        return self.rpn.loss() + self.head.loss()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

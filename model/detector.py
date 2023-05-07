import torch
import torch.nn as nn

from utils import nms
from networks import RPN, RoIPooling
from backbone import ResNet


class Detector(nn.Module):
    def __init__(self, 
                backbone: nn.Module, 
                srm: nn.Module,
                rpn: nn.Module,
                roi: nn.Module,
                bilinear: nn.Module
            ):
        self.rgb_conv_layers = ResNet(block, layers, num_classes=512)
        self.noise_conv_layers = ResNet(block, layers, num_classes=512)
        self.srm_filter_layer = srm

        self.rpn_layer = RPN(in_channels=512, feat_stride=16)

        self.roi_pooling_layer = RoIPooling(pooled_height=7, 
                                            pooled_width=7, 
                                            spatial_scale=1/16)
                                            
        self.bilinear_pooling_layer = None
        self.classifier_layer = None

    def forward(self, imgs):
        # noise
        noise = self.srm_filter_layer(imgs)

        # extract features
        rgb_feature = self.rgb_conv_layers(imgs)
        noise_feature = self.noise_conv_layers(noise)

        # proposal regions of interest
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn_layer(rgb_feature)

        # ROI pooling
        feauter = torch.cat((rgb_feature, noise_feature), dim=-1)
        pooled_feature = self.roi_pooling_layer(feature, roi.view(-1, 5))

        feature = self.bilinear_pooling_layer(rgb_feature, noise_feature)
        cls_preds = self.classifier_layer(feature)

        return bbox_preds, cls_preds
        
    def train(self, data_imgs):
        for epoch in range(200):
            for step, imgs in enumerate(data_imgs):
                ret = self.forward(imgs)
                loss = None
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print("Epoch: {}\tstep: {}\tloss: {}".format(epoch, step, loss))

            if epoch % 5 == 0:
                self.save()

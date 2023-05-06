import torch
import torch.nn as nn

from utils import nms


class Detector:
    def __init__(self, 
                backbone: nn.Module, 
                srm: nn.Module,
                rpn: nn.Module,
                roi: nn.Module,
                bilinear: nn.Module
            ):
        self.rgb_conv_layers = backbone
        self.noise_conv_layers = backbone
        self.srm_filter_layer = srm
        self.rpn_layer = rpn
        self.roi_pooling_layer = None
        self.bilinear_pooling_layer = None
        self.classifier_layer = None

    def forward(self, imgs):
        noise = self.srm_filter_layer(imgs)
        rgb_feature = self.rgb_conv_layers(imgs)
        noise_feature = self.noise_conv_layers(noise)
        bboxes, scores = self.rpn_layer(rgb_feature)

        keep = nms(bboxes, scores)
        bboxes, scores = bboxes[keep], scores[keep]

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

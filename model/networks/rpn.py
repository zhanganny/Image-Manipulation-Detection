import torch
import torch.nn as nn
import torch.nn.functional as F


class RPN(nn.Module):
    def __init__(self, in_channels, feat_stride=16):
        """
            feat_stride: backbone 的下采样压缩比例
        """
        super(RPN, self).__init__()

        self.in_channels = in_channels
        self.feat_stride = feat_stride

        self.anchor_scales = [8, 16, 32]
        self.anchor_ratios = [0.5, 1, 2]
        self.anchor_num = len(self.anchor_scales) * len(self.anchor_ratios)

        self.cls_out_channels = self.anchor_num * 2
        self.reg_out_channels = self.anchor_num * 4

        # RPN 3*3卷积
        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=512,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True)

        # Anchor 分类器: 每个Anchor输出2维概率分布
        self.classifier_layer = nn.Conv2d(in_channels=512,
                                          out_channels=self.cls_out_channels,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0)

        # Anchor 偏移回归：每个Anchor输出4维偏移
        self.bbox_regression_layer = nn.Cnov2d(in_channels=512,
                                               out_channels=self.reg_out_channels,
                                               kernel_size=1,
                                               stride=1,
                                               padding=0)

        # Proposal 
        self.proposal_layer = ProposalLayer(self.feat_stride, 
                                            self.anchor_scales,
                                            self.anchor_ratios)

    def forward(self, feature, im_info):
        """
            -feature: backbone的结果
            -im_info: P*Q -> M*N [M, N, scale_factor]
        """
        feature = F.relu(self.conv(feature), inplace=True)  # inplace 提升效率

        # classification stream
        scores = self.classifier_layer(feature)
        scores = cls.view(scores.size[0], 2, -1, scores.size[3])
        scores = F.softmax(scores, dim=1)
        scores = cls.view(scores.size[0], self.anchor_num * 2, -1, scores.size[3])

        # regression stream
        bboxes = self.bbox_regression_layer(feature)

        # proposal
        rois = self.proposal_layer(scores, bbox, im_info)

        return rois


class ProposalLayer(nn.Module):
    def __init__(self, feat_stride, anchor_scales, anchor_ratios):
        super(ProposalLayer, self).__init__()

        self.pre_nms_topN  = 20
        self.after_nms_topN = 10
        self.nms_thresh    = 0.7
        self.min_size      = 5

        self.feat_stride = feat_stride
        self.anchors = None
        self.num_amchors = self.anchors.size(0)

    def forward(self, scores, bbox_deltas, im_info):
        # 对每个位置(H, W)生成以其为中心的 Anchor Boxes

        # 剔除过小的框
        # 按 score 排序所有候选框
        # 挑选前 pre_nms_topN 个候选框
        # NMS
        # 挑选前 after_nms_topN 个候选框
        pass

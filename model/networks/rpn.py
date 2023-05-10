import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import nms

import numpy as np
from utils import _enumerate_shifted_anchor, \
                  generate_anchor_base, \
                  loc2bbox, \
                  bbox_match


# class RPN(nn.Module):
#     def __init__(self, in_channels, feat_stride=16):
#         """
#             feat_stride: backbone 的下采样压缩比例
#         """
#         super(RPN, self).__init__()

#         self.in_channels = in_channels
#         self.feat_stride = feat_stride

#         self.anchor_scales = [8, 16, 32]
#         self.anchor_ratios = [0.5, 1, 2]
#         self.anchor_num = len(self.anchor_scales) * len(self.anchor_ratios)

#         self.cls_out_channels = self.anchor_num * 2
#         self.reg_out_channels = self.anchor_num * 4

#         # RPN 3*3卷积
#         self.conv_layer = nn.Conv2d(in_channels=in_channels,
#                                     out_channels=512,
#                                     kernel_size=3,
#                                     stride=1,
#                                     padding=1,
#                                     bias=True)

#         # Anchor 分类器: 每个Anchor输出2维概率分布
#         self.classifier_layer = nn.Conv2d(in_channels=512,
#                                           out_channels=self.cls_out_channels,
#                                           kernel_size=1,
#                                           stride=1,
#                                           padding=0)

#         # Anchor 偏移回归：每个Anchor输出4维偏移
#         self.bbox_regression_layer = nn.Cnov2d(in_channels=512,
#                                                out_channels=self.reg_out_channels,
#                                                kernel_size=1,
#                                                stride=1,
#                                                padding=0)

#         # Proposal 
#         self.proposal_layer = ProposalLayer(self.feat_stride, 
#                                             self.anchor_scales,
#                                             self.anchor_ratios)

#     def forward(self, feature, im_info):
#         """
#             -feature: backbone的结果
#             -im_info: P*Q -> M*N [M, N, scale_factor]
#         """
#         feature = F.relu(self.conv(feature), inplace=True)  # inplace 提升效率

#         # classification stream
#         scores = self.classifier_layer(feature)
#         scores = cls.view(scores.size[0], 2, -1, scores.size[3])
#         scores = F.softmax(scores, dim=1)
#         scores = cls.view(scores.size[0], self.anchor_num * 2, -1, scores.size[3])

#         # regression stream
#         bboxes = self.bbox_regression_layer(feature)

#         # proposal
#         rois = self.proposal_layer(scores, bbox, im_info)

#         return rois


# class ProposalLayer(nn.Module):
#     def __init__(self, feat_stride, anchor_scales, anchor_ratios):
#         super(ProposalLayer, self).__init__()

#         self.pre_nms_topN  = 20
#         self.after_nms_topN = 10
#         self.nms_thresh    = 0.7
#         self.min_size      = 5

#         self.feat_stride = feat_stride
#         self.anchors = None
#         self.num_amchors = self.anchors.size(0)

#     def forward(self, scores, bbox_deltas, im_info):
#         # 对每个位置(H, W)生成以其为中心的 Anchor Boxes

#         # 剔除过小的框
#         # 按 score 排序所有候选框
#         # 挑选前 pre_nms_topN 个候选框
#         # NMS
#         # 挑选前 after_nms_topN 个候选框
#         pass
# from torch.nn import functional as F
# from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
# from utils.utils_bbox import loc2bbox


class RPN(nn.Module):
    def __init__(
            self, 
            in_channels=512, 
            mid_channels=512, 
            anchor_ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], 
            feat_stride=16,
            mode="training",
        ):
        super(RPN, self).__init__()
        self.mode = mode
        #-----------------------------------------#
        #   生成基础先验框，shape为[9, 4]
        #-----------------------------------------#
        self.anchor_base = generate_anchor_base(
                            anchor_scales=anchor_scales, 
                            ratios=anchor_ratios
                           )
        n_anchor = self.anchor_base.shape[0]

        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        #-----------------------------------------#
        self.conv1  = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体
        #-----------------------------------------#
        self.score  = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        #-----------------------------------------#
        #   回归预测对先验框进行调整
        #-----------------------------------------#
        self.loc    = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        #-----------------------------------------#
        #   特征点间距步长
        #-----------------------------------------#
        self.feat_stride    = feat_stride
        #-----------------------------------------#
        #   用于对建议框解码并进行非极大抑制
        #-----------------------------------------#
        self.proposal_layer = ProposalCreator(mode)
        #--------------------------------------#
        #   对FPN的网络部分进行权值初始化
        #--------------------------------------#
        # normal_init(self.conv1, 0, 0.01)
        # normal_init(self.score, 0, 0.01)
        # normal_init(self.loc, 0, 0.01)

        self.smoothL1Loss = nn.SmoothL1Loss(reduction='mean')
        self.crossEntropyLoss = nn.BCELoss(reduction='mean')
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    def forward(self, x, img_size, scale=1., annotations=None):
        n, _, h, w = x.shape

        # step 1: RPN 预测与回归
        #  预处理: 3*3 conv 特征整合
        x = F.relu(self.conv1(x)) 

        #  分支1: 1*1 conv 回归 bbox 相对于先验框 anchor 框的偏移  -> [[dx, dy, dw, dh], ...]
        rpn_locs = self.loc(x) 
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        #  分支2: 1*1 conv + softmax 预测正负样本 -> [[P(true), P(false)], ...]
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)        
        rpn_softmax_scores  = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)

        # step 2: 下面是获得先验框
        #   生成先验框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
        # 生成 anchor [[x1, y1, x2, y2], ...]
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)

        # step 3: 利用建议框网络的预测结果对先验框进行调整，调整后会对调整后的先验框进行筛选获得最终的建议框
        rois        = list()
        scores      = list()
        roi_indices = list()    # 指明 roi 属于 batch 里哪一个样本
        for i in range(n):
            roi, score = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)
            # batch_index = i * torch.ones((len(roi),))  
            batch_index = [i] * len(roi)
            rois.append(roi)
            scores.append(score)
            roi_indices += batch_index

        rois = torch.cat(rois, dim=0).type_as(x)
        scores = torch.cat(scores, dim=0).unsqueeze(1)
        # roi_indices = torch.cat(roi_indices, dim=0).type_as(x)
        roi_indices = torch.Tensor(roi_indices).unsqueeze(1)
        anchor = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)
        # 之后会用到这个建议框对共享特征层进行截取，截取之后进行roi pooling的操作，把大小固定到一样的shape上

        # 训练时计算 RPN Loss
        if self.mode == 'training': 
            assert annotations is not None
            # 每个roi的标签，有效roi下标，正样本roi下标
            labels, valid_indices, pos_indices = \
                bbox_match(rois, annotations, neg_thres=0.3, pos_thres=0.7)

            self.rpn_loss_box = self.smoothL1Loss(rois[pos_indices], annotations[0]) if len(pos_indices) > 0 else 0
            self.rpn_loss_cls = self.crossEntropyLoss(scores[valid_indices], labels[valid_indices])
            
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

    def zero_loss(self):
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    def loss(self, lamb=10):
        return self.rpn_loss_cls + lamb * self.rpn_loss_box

    def dump(self, path):
        pass


class ProposalCreator():
    def __init__(
        self, 
        mode='training', 
        nms_iou=0.7,
        n_train_pre_nms=10,
        n_train_post_nms=5,
        n_test_pre_nms=50,
        n_test_post_nms=10,
        min_size=16
    ):
        #   设置预测还是训练
        self.mode = mode
        #   建议框非极大抑制的iou大小
        self.nms_iou = nms_iou
        #   训练用到的建议框数量
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        #   预测用到的建议框数量
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # step 1: 为每个位置 (H, W) 生成 Bounding Box
        anchor = torch.from_numpy(anchor).type_as(loc)
        roi = loc2bbox(anchor, loc)
        
        # step 2: 裁剪 Bounding Box 超出图像边缘的部分
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[0])
        
        # step 3: 剔除边长过小的 Bounding Box
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # step 4: 取置信度前 n_pre_nms 高的候选框
        order = torch.argsort(score, descending=True)
        if len(order) > n_pre_nms:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # step 5: 非极大值抑制
        keep = nms(roi, score, self.nms_iou)
        
        # step 6: 取置信度前 n_post_nms 高的候选框
        if len(keep) < n_post_nms:
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
            keep = torch.cat([keep, keep[index_extra]])
        keep = keep[:n_post_nms]
        roi = roi[keep]
        score = score[keep]

        return roi, score

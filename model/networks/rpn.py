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
from torch.nn import functional as F
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils_bbox import loc2bbox

class RPN(nn.Module):
    def __init__(
        self, 
        in_channels     = 512, 
        mid_channels    = 512, 
        ratios          = [0.5, 1, 2],
        anchor_scales   = [8, 16, 32], 
        feat_stride     = 16,
        mode            = "training",
    ):
        super(RPN, self).__init__()
        #-----------------------------------------#
        #   生成基础先验框，shape为[9, 4]
        #-----------------------------------------#
        self.anchor_base    = generate_anchor_base(anchor_scales = anchor_scales, ratios = ratios)
        n_anchor            = self.anchor_base.shape[0]

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
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape
        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        #-----------------------------------------#
        x = F.relu(self.conv1(x))
        #-----------------------------------------#
        #   回归预测对先验框进行调整
        #-----------------------------------------#
        rpn_locs = self.loc(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体
        #-----------------------------------------#
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        
         #--------------------------------------------------------------------------------------#
        #   进行softmax概率计算，每个先验框只有两个判别结果
        #   内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        #--------------------------------------------------------------------------------------#
        rpn_softmax_scores  = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores       = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores       = rpn_fg_scores.view(n, -1)
        # step1 上面是获得建议框网络的生成结果
        # step2 下面是获得先验框
        #------------------------------------------------------------------------------------------------#
        #   生成先验框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
        #------------------------------------------------------------------------------------------------#
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)
        # step3 利用建议框网络的预测结果对先验框进行调整，调整后会对调整后的先验框进行筛选获得最终的建议框
        rois        = list()
        roi_indices = list()
        for i in range(n):
            roi         = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale = scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi.unsqueeze(0))
            roi_indices.append(batch_index.unsqueeze(0))

        rois        = torch.cat(rois, dim=0).type_as(x)
        roi_indices = torch.cat(roi_indices, dim=0).type_as(x)
        anchor      = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)
        # 之后会用到这个建议框对共享特征层进行截取，截取之后进行roi pooling的操作，把大小固定到一样的shape上
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


class ProposalCreator():
    def __init__(
        self, 
        mode, 
        nms_iou             = 0.7,
        n_train_pre_nms     = 12000,
        n_train_post_nms    = 600,
        n_test_pre_nms      = 3000,
        n_test_post_nms     = 300,
        min_size            = 16
    
    ):
        #-----------------------------------#
        #   设置预测还是训练
        #-----------------------------------#
        self.mode               = mode
        #-----------------------------------#
        #   建议框非极大抑制的iou大小
        #-----------------------------------#
        self.nms_iou            = nms_iou
        #-----------------------------------#
        #   训练用到的建议框数量
        #-----------------------------------#
        self.n_train_pre_nms    = n_train_pre_nms
        self.n_train_post_nms   = n_train_post_nms
        #-----------------------------------#
        #   预测用到的建议框数量
        #-----------------------------------#
        self.n_test_pre_nms     = n_test_pre_nms
        self.n_test_post_nms    = n_test_post_nms
        self.min_size           = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms   = self.n_train_pre_nms
            n_post_nms  = self.n_train_post_nms
        else:
            n_pre_nms   = self.n_test_pre_nms
            n_post_nms  = self.n_test_post_nms

        #-----------------------------------#
        #   将先验框转换成tensor
        #-----------------------------------#
        anchor = torch.from_numpy(anchor).type_as(loc)
        #-----------------------------------#
        #   将RPN网络预测结果转化成建议框
        #-----------------------------------#
        roi = loc2bbox(anchor, loc)
        #-----------------------------------#
        #   防止建议框超出图像边缘
        #-----------------------------------#
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[0])
        
        #-----------------------------------#
        #   建议框的宽高的最小值不可以小于16
        #-----------------------------------#
        min_size    = self.min_size * scale
        keep        = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        #-----------------------------------#
        #   将对应的建议框保留下来
        #-----------------------------------#
        roi         = roi[keep, :]
        score       = score[keep]

        #-----------------------------------#
        #   根据得分进行排序，取出建议框
        #-----------------------------------#
        order       = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order   = order[:n_pre_nms]
        roi     = roi[order, :]
        score   = score[order]

        #-----------------------------------#
        #   对建议框进行非极大抑制
        #   使用官方的非极大抑制会快非常多
        #-----------------------------------#
        keep    = nms(roi, score, self.nms_iou)
        if len(keep) < n_post_nms:
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
            keep        = torch.cat([keep, keep[index_extra]])
        keep    = keep[:n_post_nms]
        roi     = roi[keep]
        # 筛选后的建议框
        return roi

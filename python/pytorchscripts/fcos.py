import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from .utils import multi_apply, normal_init, constant_init
from .mobilenetv2 import InvertedResidual

class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True))


class FPN(nn.Module):

    def __init__(self, feature_layers, width_mult):
        super(FPN, self).__init__()
        agg_size = int(256 * width_mult)
        layers = []
        for i, c in enumerate(feature_layers):
            layers.append(ConvBNReLU(c, agg_size, 1, 1))
        self.fpn = nn.Sequential(*layers)
        upsample_layers = [
            nn.ConvTranspose2d(agg_size, agg_size, 2, stride=2, padding=0, dilation=1, bias=False),
            nn.ConvTranspose2d(agg_size, agg_size, 2, stride=2, padding=0, dilation=1, bias=False)
        ]
        self.upsample = nn.Sequential(*upsample_layers)
        self.top_blocks = nn.ModuleList()
        self.top_blocks.add_module('p1', nn.Conv2d(agg_size * 2, agg_size, 1, 1, bias=False))
        self.top_blocks.add_module('p2', nn.Conv2d(agg_size * 2, agg_size, 1, 1, bias=False))
        self.top_blocks.add_module('p4', nn.Conv2d(agg_size, agg_size, 3, 2, 1))
        self.top_blocks.add_module('p5', nn.Conv2d(agg_size, agg_size, 3, 2, 1))

    def forward(self, x):
        feats = []
        for idx, xx in enumerate(x):
            feats.append(self.fpn[idx](xx))
        feats_2_1 = self.upsample[1](feats[2])
        feats_1_0 = self.upsample[0](feats[1])
        fpn1 = torch.cat([feats[0], feats_1_0], 1)
        fpn2 = torch.cat([feats[1], feats_2_1], 1)
        if self.top_blocks is not None:
            p1 = self.top_blocks.p1(fpn1)
            p2 = self.top_blocks.p2(fpn2)
            p4 = self.top_blocks.p4(feats[2])
            p5 = self.top_blocks.p5(F.relu(p4))
        results = [p1, p2, feats[2], p4, p5]

        return results

    def init_weights(self):
        for up in self.upsample:
            nn.init.constant_(up.weight, 1.)
        for module in self.fpn:
            nn.init.kaiming_uniform_(module[0].weight, a=1)
        for module in self.top_blocks:
            nn.init.kaiming_uniform_(module.weight, a=1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class FPN_M(nn.Module):

    def __init__(self, out_channel, feature_layers, width_mult):
        super(FPN_M, self).__init__()

        self.channel = int(out_channel * width_mult)

        conv_layers     = [
            InvertedResidual(feature_layers[2], feature_layers[1], 1, expand_ratio=2), # stride=16
            InvertedResidual(feature_layers[1], feature_layers[0], 1, expand_ratio=2), # stride=8
            ConvBNReLU(feature_layers[0], self.channel, 1, 1)
        ]
        upsample_layers = [
            nn.ConvTranspose2d(feature_layers[1], feature_layers[1], 2, stride=2, padding=0, dilation=1, bias=False),
            nn.ConvTranspose2d(feature_layers[0], feature_layers[0], 2, stride=2, padding=0, dilation=1, bias=False)
        ]
        self.upsample = nn.Sequential(*upsample_layers)
        self.conv     = nn.Sequential(*conv_layers)

    def forward(self, x):
        out = self.conv[0](x[-1])
        out = self.upsample[0](out) + x[1]
        out = self.conv[1](out)
        out = self.upsample[1](out) + x[0]
        out = self.conv[2](out)

        results = [out]
        return results

    def init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class RPN(nn.Module):

    def __init__(self, channels, width_mult, num_classes, strides, max_joints):
        super(RPN, self).__init__()
        self.channels = int(channels * width_mult)
        self.num_classes = num_classes - 1
        cls_tower = []
        bbox_tower = []
        for i in range(4):
            cls_tower.append(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False))
            cls_tower.append(nn.GroupNorm(32, self.channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False))
            bbox_tower.append(nn.GroupNorm(32, self.channels))
            bbox_tower.append(nn.ReLU())
        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)
        self.cls_logits = nn.Conv2d(self.channels, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(self.channels, max_joints * 2, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(self.channels, 1, kernel_size=3, stride=1, padding=1)
        self.occlusion = nn.Conv2d(self.channels, max_joints * 2, kernel_size=3, stride=1, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in strides])

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        cls_feat = self.cls_tower(cls_feat)
        cls_score = self.cls_logits(cls_feat)
        centerness = self.centerness(cls_feat)
        occlusion = self.occlusion(cls_feat)
        reg_feat = self.bbox_tower(reg_feat)
        bbox_pred = scale(self.bbox_pred(reg_feat)).float()
        return cls_score, bbox_pred, centerness, occlusion

    def init_weights(self):
        normal_init(self.cls_tower[0], std=0.01)
        normal_init(self.cls_tower[3], std=0.01)
        normal_init(self.cls_tower[6], std=0.01)
        normal_init(self.cls_tower[9], std=0.01)
        normal_init(self.bbox_tower[0], std=0.01)
        normal_init(self.bbox_tower[3], std=0.01)
        normal_init(self.bbox_tower[6], std=0.01)
        normal_init(self.bbox_tower[9], std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_logits, std=0.01, bias=bias_cls)
        normal_init(self.bbox_pred, std=0.01)
        normal_init(self.centerness, std=0.01)
        normal_init(self.occlusion, std=0.01, bias=bias_cls)


class RPN_M(nn.Module):

    def __init__(self, channels, width_mult, num_classes, strides, max_joints):
        super(RPN_M, self).__init__()
        self.channels = int(channels * width_mult)
        self.num_classes = num_classes - 1
        tower = []
        for i in range(4):
            tower.append(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False))
            tower.append(nn.BatchNorm2d(self.channels))
            tower.append(nn.ReLU())
        self.tower = nn.Sequential(*tower)
        self.cls_logits = nn.Conv2d(self.channels, self.num_classes, kernel_size=1, stride=1, padding=0)
        self.bbox_pred = nn.Conv2d(self.channels, max_joints * 2, kernel_size=1, stride=1, padding=0)
        self.centerness = nn.Conv2d(self.channels, 1, kernel_size=1, stride=1, padding=0)
        self.occlusion = nn.Conv2d(self.channels, max_joints * 2, kernel_size=1, stride=1, padding=0)
        self.scales = nn.ModuleList([Scale(1.0) for _ in strides])

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        feat = x
        feat = self.tower(feat)
        cls_score  = self.cls_logits(feat)
        centerness = self.centerness(feat)
        bbox_pred  = scale(self.bbox_pred(feat)).float()
        occlusion = self.occlusion(feat)
        return cls_score, bbox_pred, centerness, occlusion

    def init_weights(self):
        normal_init(self.tower[0], std=0.01)
        normal_init(self.tower[3], std=0.01)
        normal_init(self.tower[6], std=0.01)
        normal_init(self.tower[9], std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_logits, std=0.01, bias=bias_cls)
        normal_init(self.bbox_pred, std=0.01)
        normal_init(self.centerness, std=0.01)
        normal_init(self.occlusion, std=0.01, bias=bias_cls)


class FCOSHead(nn.Module):

    def __init__(self, channels, num_classes):
        super(FCOSHead, self).__init__()
        self.channels = channels
        self.num_classes = num_classes - 1
        self.strides=(8, 16, 32, 64, 128)
        cls_tower = []
        bbox_tower = []
        for i in range(4):
            cls_tower.append(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False))
            cls_tower.append(nn.GroupNorm(32, self.channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False))
            bbox_tower.append(nn.GroupNorm(32, self.channels))
            bbox_tower.append(nn.ReLU())
        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)
        self.cls_logits = nn.Conv2d(self.channels, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(self.channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(self.channels, 1, kernel_size=3, stride=1, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        cls_feat = self.cls_tower(cls_feat)
        cls_score = self.cls_logits(cls_feat)
        centerness = self.centerness(cls_feat)
        reg_feat = self.bbox_tower(reg_feat)
        bbox_pred = scale(self.bbox_pred(reg_feat)).float().exp()
        return cls_score, bbox_pred, centerness

    def init_weights(self):
        normal_init(self.cls_tower[0], std=0.01)
        normal_init(self.cls_tower[3], std=0.01)
        normal_init(self.cls_tower[6], std=0.01)
        normal_init(self.cls_tower[9], std=0.01)
        normal_init(self.bbox_tower[0], std=0.01)
        normal_init(self.bbox_tower[3], std=0.01)
        normal_init(self.bbox_tower[6], std=0.01)
        normal_init(self.bbox_tower[9], std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_logits, std=0.01, bias=bias_cls)
        normal_init(self.bbox_pred, std=0.01)
        normal_init(self.centerness, std=0.01)


def bias_init_with_prob(prior_prob):
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

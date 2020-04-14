from torch import nn
from .l2norm import L2Norm
from .mnasnet import MnasNetA1
from .fcos import FPN, RPN, FPN_M, RPN_M


class MnasNetA1FCOS(nn.Module):

    def __init__(self,
                 num_classes=2,
                 width_mult=1.0,
                 feature_layers=[6, 12, 15],
                 strides=[8, 16, 32, 64, 128],
                 mod_fpn_rpn=False,
                 sa=True,
                 max_joints=4,
                 use_l2norm=False
                ):
        super(MnasNetA1FCOS, self).__init__()
        alpha = 0.5 if width_mult < 1.0 else 1.0
        self.num_classes = num_classes
        self.use_l2norm = use_l2norm
        self.feature_layers = feature_layers
        base = MnasNetA1.make_layers(width_mult, sa=sa)
        self.backbone = nn.ModuleList(base[:-1])
        feature_channels = []
        for i in self.feature_layers:
            feature_channels.append(self.backbone[i].conv[4].num_features)

        if mod_fpn_rpn:
            self.fpn = FPN_M(128, feature_channels, alpha)
            self.rpn = RPN_M(128, alpha, num_classes, strides, max_joints)
        else:
            self.fpn = FPN(feature_channels, alpha)
            self.rpn = RPN(256, alpha, num_classes, strides, max_joints)

        if self.use_l2norm:
            self.l2norm = L2Norm(int(32 * width_mult))

        self.init_weights()

    def forward(self, x):
        sources = []
        for i, v in enumerate(self.backbone):
            x = v(x)
            if i in self.feature_layers:
                sources.append(x)

        if self.use_l2norm:
            sources[0] = self.l2norm(sources[0])

        fpn_feats = self.fpn(sources)
        cls_score, bbox_pred, centerness, occlusion = self.rpn(fpn_feats)
        return cls_score, bbox_pred, centerness, occlusion

    def init_weights(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        self.fpn.init_weights()
        self.rpn.init_weights()

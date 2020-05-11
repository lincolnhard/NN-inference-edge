import torch
from torch import nn
import torch.nn.functional as F
from .espnetv2 import EESPNet
from .seg import EfficientPyrPool, EfficientPWConv
from .fcos import RPN_M


class ESPNetV2Fusion(nn.Module):

    def __init__(self, channels_in=3, width_mult=1.0,
                 reps_at_each_level=[0, 3, 7, 3], recept_limit=[13, 11, 9, 7, 5], branches=4,
                 num_classes_seg=20, num_classes_fcos=2,
                 strides=[8, 16, 32, 64, 128], max_joints=4):
        super().__init__()

        self.base_net = EESPNet(num_classes=num_classes_seg, channels_in=channels_in, width_mult=width_mult,
                                reps_at_each_level=reps_at_each_level, recept_limit=recept_limit, branches=branches)
        del self.base_net.classifier
        del self.base_net.level5
        del self.base_net.level5_0
        config = self.base_net.config

        base_dec_planes = 16
        dec_planes = [4 * base_dec_planes, 3 * base_dec_planes, 2 * base_dec_planes, num_classes_seg]
        pyr_plane_proj = min(num_classes_seg // 2, base_dec_planes)

        self.bu_dec_l1 = EfficientPyrPool(in_planes=config[3], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[0])
        self.bu_dec_l2 = EfficientPyrPool(in_planes=dec_planes[0], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[1])
        self.bu_dec_l3 = EfficientPyrPool(in_planes=dec_planes[1], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[2])
        self.bu_dec_l4 = EfficientPyrPool(in_planes=dec_planes[2], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[3], last_layer_br=False)

        self.merge_enc_dec_l2 = EfficientPWConv(config[2], dec_planes[0])
        self.merge_enc_dec_l3 = EfficientPWConv(config[1], dec_planes[1])
        self.merge_enc_dec_l4 = EfficientPWConv(config[0], dec_planes[2])

        self.bu_br_l2 = nn.Sequential(nn.BatchNorm2d(dec_planes[0]), nn.PReLU(dec_planes[0]))
        self.bu_br_l3 = nn.Sequential(nn.BatchNorm2d(dec_planes[1]), nn.PReLU(dec_planes[1]))
        self.bu_br_l4 = nn.Sequential(nn.BatchNorm2d(dec_planes[2]), nn.PReLU(dec_planes[2]))

        alpha = 0.5 if width_mult < 1.0 else 1.0
        self.rpn = RPN_M(48, alpha, num_classes_fcos, strides, max_joints)

        #self.upsample =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_layers = nn.ModuleList()
        for i in dec_planes[:3]:
            self.upsample_layers.append(nn.ConvTranspose2d(i, i, kernel_size=4, stride=2, padding=1, dilation=1, bias=False))

        #self.upsample_layer = nn.ConvTranspose2d(proj_planes, proj_planes, kernel_size=4, stride=2, padding=1, dilation=1, bias=False, groups=proj_planes)

        self.init_params()
        self.rpn.init_weights()

    def upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_basenet_params(self):
        modules_base = [self.base_net]
        for i in range(len(modules_base)):
            for m in modules_base[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_segment_params(self):
        modules_seg = [self.bu_dec_l1, self.bu_dec_l2, self.bu_dec_l3, self.bu_dec_l4,
                       self.merge_enc_dec_l4, self.merge_enc_dec_l3, self.merge_enc_dec_l2,
                       self.bu_br_l4, self.bu_br_l3, self.bu_br_l2]
        for i in range(len(modules_seg)):
            for m in modules_seg[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_rpn_params(self):
        modules_rpn = [self.rpn]
        for i in range(len(modules_rpn)):
            for m in modules_rpn[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.ReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def forward(self, x):
        x_size = (x.size(2), x.size(3))
        enc_out_l1 = self.base_net.level1(x)
        enc_out_l2 = self.base_net.level2_0(2, enc_out_l1, x)
        enc_out_l3_0 = self.base_net.level3_0(3, enc_out_l2, x)
        enc_out_l3 = enc_out_l3_0
        for _, layer in enumerate(self.base_net.level3):
            enc_out_l3 = layer(enc_out_l3)

        enc_out_l4_0 = self.base_net.level4_0(4, enc_out_l3, x)
        enc_out_l4 = enc_out_l4_0
        for _, layer in enumerate(self.base_net.level4):
            enc_out_l4 = layer(enc_out_l4)

        # bottom-up decoding
        bu_out = self.bu_dec_l1(enc_out_l4)

        # Decoding block
        bu_out = self.upsample_layers[0](bu_out)
        enc_out_l3_proj = self.merge_enc_dec_l2(enc_out_l3)
        bu_out = enc_out_l3_proj + bu_out
        bu_out = self.bu_br_l2(bu_out)
        bu_out = self.bu_dec_l2(bu_out)

        # RPN
        cls_score, bbox_pred, centerness, occlusion = self.rpn([bu_out])

        #decoding block
        bu_out = self.upsample_layers[1](bu_out)
        enc_out_l2_proj = self.merge_enc_dec_l3(enc_out_l2)
        bu_out = enc_out_l2_proj + bu_out
        bu_out = self.bu_br_l3(bu_out)
        bu_out = self.bu_dec_l3(bu_out)


        # decoding block
        bu_out = self.upsample_layers[2](bu_out)
        enc_out_l1_proj = self.merge_enc_dec_l4(enc_out_l1)
        bu_out = enc_out_l1_proj + bu_out
        bu_out = self.bu_br_l4(bu_out)
        bu_out  = self.bu_dec_l4(bu_out)

        return cls_score, bbox_pred, centerness, occlusion, bu_out

        # seg_out = F.interpolate(bu_out, size=x_size, mode='bilinear', align_corners=True)
        # return cls_score, bbox_pred, centerness, occlusion, seg_out

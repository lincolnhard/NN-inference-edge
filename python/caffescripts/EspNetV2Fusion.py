import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np

from .BaseBlock import CBR, DownSampler, EESP, EfficientPyrPool, EfficientPWConv, BR, ConvBNReLU
from .AttentionLayer import AttentionType
from .FillWeight import get_weight_from_ConvBNReLU, get_weight_from_MBConvBlock, get_weight_from_headblock, get_weight_from_InvertedResidual, get_weight_from_Deconvolution, pytorch_bn_conv_fuse

class ESPNetV2Fusion():
    def __init__(self,
                 width_mult=1.0,
                 reps_at_each_level=[0, 3, 7, 3],
                 recept_limit=[13, 11, 9, 7, 5],
                 branches=4,
                 num_classes_seg=20,
                 num_classes_fcos=3,
                 strides=[8, 16, 32, 64, 128],
                 max_joints=4
                ):
        super(ESPNetV2Fusion, self).__init__()
        self.width_mult = width_mult
        self.reps_at_each_level = reps_at_each_level
        self.recept_limit = recept_limit
        self.branches = branches
        self.num_classes_seg = num_classes_seg
        self.num_classes_fcos = num_classes_fcos
        self.strides = strides
        self.max_joints = max_joints

    
    def createPrototxt(self, net):
        layer_idx = 0
        ALPHA_MAP = {
            0.5: [16, 32, 64, 128, 256, 1024],
            1.0: [32, 64, 128, 256, 512, 1024],
            1.25: [32, 80, 160, 320, 640, 1024],
            1.5: [32, 96, 192, 384, 768, 1024],
            2.0: [32, 128, 256, 512, 1024, 1280]
        }

        if not self.width_mult in ALPHA_MAP.keys():
            print('Model at scale s={} is not suppoerted yet'.format(self.width_mult))
            exit(-1)

        out_channel_map = ALPHA_MAP[self.width_mult]
        K = [self.branches] * len(self.recept_limit)
        self.input_reinforcement = True

        base_dec_planes = 16
        dec_planes = [4 * base_dec_planes, 3 * base_dec_planes, 2 * base_dec_planes, self.num_classes_seg]
        pyr_plane_proj = min(self.num_classes_seg // 2, base_dec_planes)

        layer_idx = 0
        x = net['data']
        enc_out_l1, layer_idx = CBR(net, layer_idx, x, out_channel_map[0], kernel_size=3, stride=2)
        #TODO: hardcoded 2 here
        enc_out_l2, layer_idx = DownSampler(net, layer_idx, enc_out_l1, x, 2, out_channel_map[0], out_channel_map[1], k=K[0], r_lim=self.recept_limit[0], reinf=self.input_reinforcement)
        #TODO: hardcoded 3 here
        enc_out_l3_0, layer_idx = DownSampler(net, layer_idx, enc_out_l2, x, 3, out_channel_map[1], out_channel_map[2], k=K[1], r_lim=self.recept_limit[1], reinf=self.input_reinforcement)

        enc_out_l3 = enc_out_l3_0
        for _ in range(self.reps_at_each_level[1]):
            enc_out_l3, layer_idx = EESP(net, layer_idx, enc_out_l3, out_channel_map[2], out_channel_map[2], stride=1, k=K[2], r_lim=self.recept_limit[2])
        #TODO: hardcoded 4 here
        enc_out_l4_0, layer_idx = DownSampler(net, layer_idx, enc_out_l3, x, 4, out_channel_map[2], out_channel_map[3], k=K[2], r_lim=self.recept_limit[2], reinf=self.input_reinforcement)

        enc_out_l4 = enc_out_l4_0
        for _ in range(self.reps_at_each_level[2]):
            enc_out_l4, layer_idx = EESP(net, layer_idx, enc_out_l4, out_channel_map[3], out_channel_map[3], stride=1, k=K[3], r_lim=self.recept_limit[3])

        # bottom-up decoding
        bu_out, layer_idx = EfficientPyrPool(net, layer_idx, enc_out_l4, in_planes=out_channel_map[3], proj_planes=pyr_plane_proj, out_planes=dec_planes[0])

        # Decoding block
        net['deconv{}'.format(layer_idx)] = L.Deconvolution(bu_out,
                                                            param={"lr_mult": 1, "decay_mult": 1},
                                                            convolution_param=dict(
                                                                                num_output=dec_planes[0],
                                                                                bias_term=False,
                                                                                kernel_size=4,
                                                                                stride=2,
                                                                                pad=1,
                                                                                dilation=1
                                                                                )
                                                            )
        bu_out = net['deconv{}'.format(layer_idx)]
        layer_idx += 1

        enc_out_l3_proj, layer_idx = EfficientPWConv(net, layer_idx, enc_out_l3, out_channel_map[2], dec_planes[0], 60, 80)

        net['add{}'.format(layer_idx)] = L.Eltwise(enc_out_l3_proj, bu_out, operation=P.Eltwise.SUM, coeff=[1,1])
        bu_out = net['add{}'.format(layer_idx)]
        layer_idx += 1

        bu_out, layer_idx = BR(net, layer_idx, bu_out)
        bu_out, layer_idx = EfficientPyrPool(net, layer_idx, bu_out, in_planes=dec_planes[0], proj_planes=pyr_plane_proj, out_planes=dec_planes[1])

        # RPN
        layer_idx = self.rpn_layer(net, layer_idx, bu_out, 48, idx=0)

        #decoding block
        net['deconv{}'.format(layer_idx)] = L.Deconvolution(bu_out,
                                                            param={"lr_mult": 1, "decay_mult": 1},
                                                            convolution_param=dict(
                                                                                num_output=dec_planes[1],
                                                                                bias_term=False,
                                                                                kernel_size=4,
                                                                                stride=2,
                                                                                pad=1,
                                                                                dilation=1
                                                                                )
                                                            )
        bu_out = net['deconv{}'.format(layer_idx)]
        layer_idx += 1

        enc_out_l2_proj, layer_idx = EfficientPWConv(net, layer_idx, enc_out_l2, out_channel_map[1], dec_planes[1], 120, 160)

        net['add{}'.format(layer_idx)] = L.Eltwise(enc_out_l2_proj, bu_out, operation=P.Eltwise.SUM, coeff=[1,1])
        bu_out = net['add{}'.format(layer_idx)]
        layer_idx += 1

        bu_out, layer_idx = BR(net, layer_idx, bu_out)
        bu_out, layer_idx = EfficientPyrPool(net, layer_idx, bu_out, in_planes=dec_planes[1], proj_planes=pyr_plane_proj, out_planes=dec_planes[2])

        #decoding block
        net['deconv{}'.format(layer_idx)] = L.Deconvolution(bu_out,
                                                            param={"lr_mult": 1, "decay_mult": 1},
                                                            convolution_param=dict(
                                                                                num_output=dec_planes[2],
                                                                                bias_term=False,
                                                                                kernel_size=4,
                                                                                stride=2,
                                                                                pad=1,
                                                                                dilation=1
                                                                                )
                                                            )
        bu_out = net['deconv{}'.format(layer_idx)]
        layer_idx += 1

        enc_out_l1_proj, layer_idx = EfficientPWConv(net, layer_idx, enc_out_l1, out_channel_map[0], dec_planes[2], 240, 320)

        net['add{}'.format(layer_idx)] = L.Eltwise(enc_out_l1_proj, bu_out, operation=P.Eltwise.SUM, coeff=[1,1])
        bu_out = net['add{}'.format(layer_idx)]
        layer_idx += 1

        bu_out, layer_idx = BR(net, layer_idx, bu_out)
        bu_out, layer_idx = EfficientPyrPool(net, layer_idx, bu_out, in_planes=dec_planes[2], proj_planes=pyr_plane_proj, out_planes=dec_planes[3])





    def rpn_layer(self, net, layer_idx, bottom_blob, channel, idx):
        for _ in range(4):
            bottom_blob, layer_idx = ConvBNReLU(net, layer_idx, bottom_blob, int(channel * self.width_mult), kernel_size=3, stride=1)

        cls_score, layer_idx = ConvBNReLU(net, layer_idx,
                                          bottom_blob,
                                          self.num_classes_fcos-1,
                                          kernel_size=1,
                                          stride=1,
                                          use_activation=False,
                                          bias_term=True)
        centerness, layer_idx = ConvBNReLU(net, layer_idx,
                                           bottom_blob,
                                           1,
                                           kernel_size=1,
                                           stride=1,
                                           use_activation=False,
                                           bias_term=True)
        vetex_pred, layer_idx = ConvBNReLU(net, layer_idx,
                                          bottom_blob,
                                          self.max_joints*2,
                                          kernel_size=1,
                                          stride=1,
                                          use_activation=False,
                                          bias_term=True)
        occlusion, layer_idx = ConvBNReLU(net, layer_idx,
                                          bottom_blob,
                                          self.max_joints*2,
                                          kernel_size=1,
                                          stride=1,
                                          use_activation=False,
                                          bias_term=True)
        net['cls_score'] = L.Sigmoid(cls_score)
        net['centerness'] = L.Sigmoid(centerness)
        net['occlusion'] = L.Sigmoid(occlusion)
        net['scoremap_perm'] = L.Permute(net['cls_score'], order=[0, 2, 3, 1])
        net['centernessmap_perm'] = L.Permute(net['centerness'], order=[0, 2, 3, 1])
        net['occlusionmap_perm'] = L.Permute(net['occlusion'], order=[0, 2, 3, 1])
        net['regressionmap_perm'] = L.Permute(vetex_pred, order=[0, 2, 3, 1])

        return layer_idx
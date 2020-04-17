import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np

from .BaseBlock import ConvBNReLU, InvertedResidual, Deconvolution, MBConvBlock
from .AttentionLayer import AttentionType
from .FillWeight import get_weight_from_ConvBNReLU, get_weight_from_MBConvBlock, get_weight_from_headblock, get_weight_from_InvertedResidual, get_weight_from_Deconvolution, pytorch_bn_conv_fuse


class MnasNetA1FCOS():
    def __init__(self,
                 num_classes,
                 width_mult,
                 max_joints
                ):
        super(MnasNetA1FCOS, self).__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.max_joints = max_joints

    def createPrototxt(self, net, neth, netw):
        layer_idx = self.base_model(net, neth, netw)
        layer_idx, bottom_blob = self.fpn_layer(net, layer_idx, 128)
        layer_idx = self.rpn_layer(net, layer_idx, bottom_blob, 128, idx=0)

    def base_model(self, net, height, width):
        layer_idx = 0
        input_channel = int(32 * self.width_mult)
        # settings = [
        #     # t, c, n, s, k, r
        #     [1, 16, 1, 1, 3, 1],  # SepConv_3x3
        #     [6, 24, 2, 2, 3, 1],  # MBConv6_3x3
        #     [3, 40, 3, 2, 5, 4],  # MBConv3_5x5, SE
        #     [6, 80, 4, 2, 3, 1],  # MBConv6_3x3
        #     [6, 112, 2, 1, 3, 4],  # MBConv6_3x3, SE
        #     [6, 160, 3, 2, 5, 4],  # MBConv6_5x5, SE
        #     [6, 320, 1, 1, 3, 1]  # MBConv6_3x3
        # ]
        settings = [
            # t, c, n, s, k, r, x
            [1, 16, 1, 1, 3, 1, 0],  # SepConv_3x3
            [6, 24, 2, 2, 3, 1, 0],  # MBConv6_3x3
            [3, 40, 3, 2, 5, 4, 1],  # MBConv3_5x5, SE
            [6, 80, 4, 2, 3, 1, 0],  # MBConv6_3x3
            [6, 112, 2, 1, 3, 4, 1],  # MBConv6_3x3, SE
            [6, 160, 3, 2, 5, 4, 1]  # MBConv6_5x5, SE
        ]
        self.fpn_list_names = []
        self.fpn_list_channels = []
        bottom_blob, layer_idx = ConvBNReLU(net, layer_idx, net['data'], input_channel, stride=2)
        height = int(height*0.5)
        width  = int(width*0.5)
        for i, (t, c, n, s, k, r, x) in enumerate(settings):
            output_channel = int(c * self.width_mult)
            no_skip = True if i == 0 else False
            for j in range(n):
                stride = s if j == 0 else 1
                if stride > 1:
                    height = round((height+0.5)/stride)
                    width  = round((width+0.5)/stride)
                bottom_blob, layer_idx = MBConvBlock(net, layer_idx,
                                                     bottom_blob,
                                                     input_channel,
                                                     output_channel,
                                                     stride,
                                                     t,
                                                     k,
                                                     r,
                                                     no_skip,
                                                     height,
                                                     width,
                                                     attentionType=AttentionType.SPATIAL)
                input_channel = output_channel

            if x == 1:
                self.fpn_list_names.append(list(net.tops.items())[-1][0])
                self.fpn_list_channels.append(output_channel)

        return layer_idx

    def fpn_layer(self, net, layer_idx, out_channel):
        out_channel = int(out_channel * self.width_mult)
        bottom_blob, layer_idx = InvertedResidual(net,
                                                  layer_idx,
                                                  net[self.fpn_list_names[2]],
                                                  self.fpn_list_channels[2],
                                                  self.fpn_list_channels[1],
                                                  1,
                                                  2)
        bottom_blob, layer_idx = Deconvolution(net,
                                               layer_idx,
                                               bottom_blob,
                                               self.fpn_list_channels[1],
                                               skip_connect=net[self.fpn_list_names[1]])
        bottom_blob, layer_idx = InvertedResidual(net,
                                                  layer_idx,
                                                  bottom_blob,
                                                  self.fpn_list_channels[1],
                                                  self.fpn_list_channels[0],
                                                  1,
                                                  2)
        bottom_blob, layer_idx = Deconvolution(net,
                                               layer_idx,
                                               bottom_blob,
                                               self.fpn_list_channels[0],
                                               skip_connect=net[self.fpn_list_names[0]])
        bottom_blob, layer_idx = ConvBNReLU(net, layer_idx, bottom_blob, out_channel, kernel_size=1, stride=1)
        return layer_idx, bottom_blob

    def rpn_layer(self, net, layer_idx, bottom_blob, channel, idx):
        for i in range(4):
            bottom_blob, layer_idx = ConvBNReLU(net, layer_idx, bottom_blob, int(channel * self.width_mult), kernel_size=3, stride=1)

        cls_score, layer_idx = ConvBNReLU(net, layer_idx,
                                          bottom_blob,
                                          self.num_classes-1,
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

    def storeWeighting(self, globallist ,pytorchModel):
        # features
        for modu in pytorchModel.backbone:
            if modu.__class__.__name__ == 'ConvBNReLU':
                get_weight_from_ConvBNReLU(globallist, modu)
            if modu.__class__.__name__ == 'MBConvBlock':
                get_weight_from_MBConvBlock(globallist, modu)
        # fpn
        get_weight_from_InvertedResidual(globallist, pytorchModel.fpn.conv[0])
        get_weight_from_Deconvolution(globallist, pytorchModel.fpn.upsample[0])
        get_weight_from_InvertedResidual(globallist, pytorchModel.fpn.conv[1])
        get_weight_from_Deconvolution(globallist, pytorchModel.fpn.upsample[1])
        get_weight_from_ConvBNReLU(globallist, pytorchModel.fpn.conv[2])

        #rpn
        rpn_model = pytorchModel.rpn
        for idx in range(4):
            globallist.append(pytorch_bn_conv_fuse(rpn_model.tower[3*idx], rpn_model.tower[3*idx+1]))
        globallist.append(rpn_model.cls_logits)
        globallist.append(rpn_model.centerness)
        globallist.append(rpn_model.bbox_pred)
        globallist.append(rpn_model.occlusion)
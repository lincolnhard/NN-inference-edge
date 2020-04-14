import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np
from .AttentionLayer import AttentionType, SqeezeExcitationLayer, SpatialAttentionLayer

def ConvBNReLU(caffe_net, layer_idx,
               bottom_blob,
               out_planes,
               kernel_size=3,
               stride=1,
               groups=1,
               auto_padding=True,
               use_activation=True,
               bias_term=True,
               masks=None,
               unpruned=False):
    padding = (kernel_size - 1) // 2 if auto_padding else 0
    names = ['conv{}'.format(layer_idx), 'conv{}_relu'.format(layer_idx)]
    out_ch_size = out_planes
    conv_groups = groups

    # This section is for pruning
    if masks is not None:
        in_mask = masks.pop(0)
        out_mask = masks[0]
        out_mask_ch_size = len(out_mask)  # original output channel size
        assert out_mask_ch_size == out_ch_size
        if unpruned == True:
            masks[0] = np.ones(len(out_mask), dtype=np.float32)
        else:
            out_ch_size = int(np.sum(out_mask)) # pruned output channel size
        if groups != 1:
            conv_groups = int(np.sum(in_mask)) # pruned input channel size
            masks[0] = in_mask
            out_ch_size = conv_groups
    # End of section

    caffe_net[names[0]] = L.Convolution(bottom_blob,
                                        num_output=out_ch_size,
                                        bias_term=bias_term,
                                        pad=padding,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        group=conv_groups)

    if use_activation == True:
        caffe_net[names[1]] = L.ReLU(caffe_net[names[0]], in_place=True)

    return caffe_net[names[0]], layer_idx + 1


def Deconvolution(caffe_net, layer_idx,
                  bottom_blob,
                  out_planes,
                  kernel_size=2,
                  stride=2,
                  padding=0,
                  skip_connect=None
                 ):
    names = ['deconv{}'.format(layer_idx)]
    caffe_net[names[0]] = L.Deconvolution(bottom_blob,
                                  param={"lr_mult": 1, "decay_mult": 1},
                                  convolution_param=dict(
                                          num_output=out_planes,
                                          bias_term=False,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          pad=padding
                                          ))
    bottom_blob = caffe_net[names[0]]

    if skip_connect is not None:
        caffe_net['res{}'.format(layer_idx)] = L.Eltwise(bottom_blob,
                                                         skip_connect,
                                                         operation=P.Eltwise.SUM,
                                                         coeff=[1,1])
        bottom_blob = caffe_net['res{}'.format(layer_idx)]

    return bottom_blob, layer_idx + 1


def InvertedResidual(caffe_net,
                     layer_idx,
                     bottom_blob,
                     in_planes,
                     out_planes,
                     stride,
                     expand_ratio,
                     masks=None):
    hidden_dim = int(round(in_planes * expand_ratio))
    use_res_connect = stride == 1 and in_planes == out_planes
    start_bottom_blob = bottom_blob

    # pw
    if expand_ratio != 1:
        bottom_blob, layer_idx = ConvBNReLU(caffe_net, layer_idx, bottom_blob, hidden_dim, kernel_size=1, masks=masks)

    # dw
    bottom_blob, layer_idx = ConvBNReLU(caffe_net,
                                        layer_idx,
                                        bottom_blob,
                                        hidden_dim,
                                        stride=stride,
                                        groups=hidden_dim,
                                        masks=masks)

    # pw-linear
    '''
    caffe_net['conv{}'.format(layer_idx)] = L.Convolution(bottom_blob,
                                                          num_output=out_planes,
                                                          bias_term=True,
                                                          kernel_size=1,
                                                          stride=1,
                                                          pad=0)
    bottom_blob = caffe_net['conv{}'.format(layer_idx)]
    '''
    bottom_blob, layer_idx = ConvBNReLU(caffe_net,
                                        layer_idx,
                                        bottom_blob,
                                        out_planes,
                                        kernel_size=1,
                                        use_activation=False,
                                        masks=masks,
                                        unpruned=True)

    if use_res_connect:
        caffe_net['res{}'.format(layer_idx-1)] = L.Eltwise(bottom_blob,
                                                         start_bottom_blob,
                                                         operation=P.Eltwise.SUM,
                                                         coeff=[1,1])
        bottom_blob = caffe_net['res{}'.format(layer_idx-1)]

    return bottom_blob, layer_idx


def MBConvBlock(caffe_net, layer_idx,
                bottom_blob,
                in_planes,
                out_planes,
                stride,
                expand_ratio,
                kernel_size=3,
                reduction_ratio=1,
                no_skip=False,
                height=0,
                width=0,
                attentionType=AttentionType.CHANNEL,
                masks=None):
    use_res_connect = in_planes == out_planes and stride == 1 and not no_skip
    hidden_dim = int(in_planes * expand_ratio)
    start_bottom_blob = bottom_blob
    
    # pw
    if in_planes != hidden_dim:
        bottom_blob, layer_idx = ConvBNReLU(caffe_net, layer_idx, bottom_blob, hidden_dim, kernel_size=1, masks=masks)

    # dw
    bottom_blob, layer_idx = ConvBNReLU(caffe_net,
                                        layer_idx,
                                        bottom_blob,
                                        hidden_dim,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        groups=hidden_dim,
                                        masks=masks)

    if reduction_ratio != 1:
        if attentionType == AttentionType.CHANNEL:
            reduced_dim = max(1, int(in_planes / reduction_ratio))
            bottom_blob, layer_idx = SqeezeExcitationLayer(caffe_net,
                                                           layer_idx,
                                                           bottom_blob,
                                                           hidden_dim,
                                                           reduced_dim,
                                                           height,
                                                           width,
                                                           bias_term=True)

        elif attentionType == AttentionType.SPATIAL:
            bottom_blob, layer_idx = SpatialAttentionLayer(caffe_net,
                                                           layer_idx,
                                                           bottom_blob,
                                                           hidden_dim,
                                                           bias_term=False,
                                                           masks=masks)

    # pw-linear
    bottom_blob, layer_idx = ConvBNReLU(caffe_net,
                                        layer_idx,
                                        bottom_blob,
                                        out_planes,
                                        kernel_size=1,
                                        auto_padding=False,
                                        use_activation=False,
                                        masks=masks,
                                        unpruned=True)
    if use_res_connect:
        caffe_net['res{}'.format(layer_idx-1)] = L.Eltwise(bottom_blob,
                                                         start_bottom_blob,
                                                         operation=P.Eltwise.SUM,
                                                         coeff=[1,1])
        bottom_blob = caffe_net['res{}'.format(layer_idx-1)]
    
    return bottom_blob, layer_idx


def HeadBlock(caffe_net, layer_idx,
              bottom_blob,
              in_planes,
              out_planes,
              additional_conv=False,
              use_SeLyaer=False,
              height=0,
              width=0,
              attentionType=AttentionType.CHANNEL,
              masks=None):
    hidden_dim = in_planes * 2 if additional_conv else in_planes

    if additional_conv:
        if use_SeLyaer:
            bottom_blob, layer_idx = MBConvBlock(caffe_net,
                                                 layer_idx,
                                                 bottom_blob,
                                                 in_planes,
                                                 hidden_dim,
                                                 stride=1,
                                                 expand_ratio=6,
                                                 reduction_ratio=4,
                                                 height=height,
                                                 width=width,
                                                 attentionType=attentionType,
                                                 masks=masks)
        else:
            bottom_blob, layer_idx = InvertedResidual(caffe_net,
                                                      layer_idx,
                                                      bottom_blob,
                                                      in_planes,
                                                      hidden_dim,
                                                      stride=1,
                                                      expand_ratio=6)

    caffe_net['conv{}'.format(layer_idx)] = L.Convolution(bottom_blob, num_output=out_planes, kernel_size=3, pad=1)
    bottom_blob = caffe_net['conv{}'.format(layer_idx)]
    
    return bottom_blob, layer_idx + 1


# Just code test
if __name__ == '__main__':
    net = caffe.NetSpec()

    net['data'] = L.Input(shape=dict(dim=[1, 16, 360, 640]))
    #net['output'], _ = InvertedResidual(net, 0, net['data'], in_planes = 16, out_planes = 16, stride=1, expand_ratio=6)
    #net['output'], _ = MBConvBlock(net, 0, net['data'], in_planes = 16, out_planes = 16, stride=1, expand_ratio=6, reduction_ratio=4)
    net['output'], _ = HeadBlock(net, 0, net['data'], in_planes = 16, out_planes = 16, additional_conv=True, use_SeLyaer=True)

    print(str(net.to_proto()))
    #print(net.blobs['conv1'].data[0].shape)
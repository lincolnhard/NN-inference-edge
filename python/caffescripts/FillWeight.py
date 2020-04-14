import torch
from torch import nn
import numpy as np

def feed_weighting(net, weighting_list):
    
    assert len(weighting_list) == len(list(net.params))
    for (k, v), pyt in zip(net.params.items(), weighting_list):
        v[0].data[...] = pyt.weight.detach().numpy()
        if pyt.bias is not None:
            v[1].data[...] = pyt.bias.detach().numpy()

def pytorch_conv2fc(conv):
    w = conv.weight
    b = conv.bias

    w = torch.squeeze( torch.squeeze(w,3), 2)

    fc = nn.Linear(conv.in_channels, conv.out_channels)
    fc.weight = nn.Parameter(w)
    fc.bias = nn.Parameter(b)
    
    return fc

def pytorch_bn_conv_fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight
    beta = bn.bias
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    w = w * (gamma / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * gamma + beta
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         groups = conv.groups,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)

    return fused_conv


def get_weight_from_ConvBNReLU(globallist, modual):
    globallist.append(pytorch_bn_conv_fuse(modual[0], modual[1]))

def get_weight_from_InvertedResidual(globallist, modual):
    for submodual in modual.conv:
        if submodual.__class__.__name__ == 'ConvBNReLU':
            get_weight_from_ConvBNReLU(globallist, submodual)

    globallist.append(pytorch_bn_conv_fuse(modual.conv[-2], modual.conv[-1]))

def get_weight_from_Deconvolution(globallist, modual):
    if modual.__class__.__name__ == 'ConvTranspose2d':
        globallist.append(modual)

def get_weight_from_SqueezeExcitation(globallist, modual):
    for submodual in modual.se:
        if submodual.__class__.__name__ == 'Conv2d':
            globallist.append(pytorch_conv2fc(submodual))


def get_weight_from_SpatialAttentionLayer(globallist, modual):
    conv = modual.sa[0]

    squeeze_filt = torch.Tensor( np.ones( (1, conv.out_channels, 1, 1) ) / conv.out_channels)
    
    sqz_conv = nn.Conv2d(conv.out_channels,
                         1,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         groups = conv.groups,
                         bias=False)
    sqz_conv.weight = nn.Parameter(squeeze_filt)
    globallist.append(sqz_conv)
    globallist.append(conv)

def get_weight_from_MBConvBlock(globallist, modual):
    for submodual in modual.conv:
        if submodual.__class__.__name__ == 'ConvBNReLU':
            get_weight_from_ConvBNReLU(globallist, submodual)
        elif submodual.__class__.__name__ == 'SqueezeExcitation':
            get_weight_from_SqueezeExcitation(globallist, submodual)
        elif submodual.__class__.__name__ == 'SpatialAttention':
            get_weight_from_SpatialAttentionLayer(globallist, submodual)

    globallist.append(pytorch_bn_conv_fuse(modual.conv[-2], modual.conv[-1]))

def get_weight_from_headblock(globallist, modual):
    if len(modual) == 2:
        if modual[0].__class__.__name__ == 'InvertedResidual':
            get_weight_from_InvertedResidual(globallist, modual[0])
        elif modual[0].__class__.__name__ == 'MBConvBlock':
            get_weight_from_MBConvBlock(globallist, modual[0])

        globallist.append(modual[1])

    elif len(modual) == 1:
        globallist.append(modual[0])
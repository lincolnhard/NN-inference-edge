import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np
from .AttentionLayer import AttentionType, SqeezeExcitationLayer, SpatialAttentionLayer
import math

def BR(
        caffe_net,
        layer_idx,
        bottom_blob
        ):

    names = ['bn{}'.format(layer_idx), 'prelu{}'.format(layer_idx)]

    caffe_net[names[0]] = L.BatchNorm(bottom_blob)
    caffe_net[names[1]] = L.PReLU(caffe_net[names[0]])
    return caffe_net[names[1]], layer_idx + 1
    

def CB(
        caffe_net,
        layer_idx,
        bottom_blob,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        ):

    padding = int((kernel_size - 1) / 2)

    names = ['conv{}'.format(layer_idx)]

    caffe_net[names[0]] = L.Convolution(bottom_blob,
                                        num_output=out_planes,
                                        bias_term=False,
                                        pad=padding,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        group=groups)

    return caffe_net[names[0]], layer_idx + 1

def CBR(
        caffe_net,
        layer_idx,
        bottom_blob,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
       ):

    padding = int((kernel_size - 1) / 2)

    names = ['conv{}'.format(layer_idx), 'prelu{}'.format(layer_idx)]

    caffe_net[names[0]] = L.Convolution(bottom_blob,
                                        num_output=out_planes,
                                        bias_term=False,
                                        pad=padding,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        group=groups)

    caffe_net[names[1]] = L.PReLU(caffe_net[names[0]])
    return caffe_net[names[1]], layer_idx + 1

def CDilated(
            caffe_net,
            layer_idx,
            bottom_blob,
            out_planes,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1
            ):

    padding = int((kernel_size - 1) / 2) * dilation

    names = ['conv{}'.format(layer_idx)]

    caffe_net[names[0]] = L.Convolution(bottom_blob,
                                        num_output=out_planes,
                                        bias_term=False,
                                        pad=padding,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        group=groups,
                                        dilation=dilation)

    return caffe_net[names[0]], layer_idx + 1


def EESP(
        caffe_net,
        layer_idx,
        bottom_blob,
        nIn,
        nOut,
        stride=1,
        k=4,
        r_lim=7,
        down_method='esp'
        ):

    inblob = bottom_blob
    n = int(nOut / k)
    n1 = nOut - (k - 1) * n
    output1, layer_idx = CBR(caffe_net, layer_idx, bottom_blob, n, 1, 1, k)

    map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
    k_sizes = list()
    for i in range(k):
        ksize = int(3 + 2 * i)
        ksize = ksize if ksize <= r_lim else 3
        k_sizes.append(ksize)
    k_sizes.sort()
    downAvg = True if down_method == 'avg' else False

    output = []
    bblist = []
    for i in range(k):
        d_rate = map_receptive_ksize[k_sizes[i]]
        bb, layer_idx = CDilated(caffe_net, layer_idx, output1, n, 3, stride, d_rate, n)
        bblist.append(bb)
    
    bb = bblist[0]
    output.append(bb)
    for i in range(k-1):
        caffe_net['add{}'.format(layer_idx)] = L.Eltwise(bb,
                                                    bblist[i+1],
                                                    operation=P.Eltwise.SUM,
                                                    coeff=[1,1])
        output.append(caffe_net['add{}'.format(layer_idx)])
        bb = caffe_net['add{}'.format(layer_idx)]
        layer_idx += 1


    caffe_net['cat{}'.format(layer_idx)] = L.Concat(*output, axis=1)
    bottom_blob, layer_idx = BR(caffe_net, layer_idx, caffe_net['cat{}'.format(layer_idx)])
    expanded, layer_idx = CB(caffe_net, layer_idx, bottom_blob, nOut, 1, 1, k)

    if stride == 2 and downAvg:
        return expanded, layer_idx

    caffe_net['add{}'.format(layer_idx)] = L.Eltwise(inblob, expanded, operation=P.Eltwise.SUM, coeff=[1,1])

    caffe_net['prelu{}'.format(layer_idx)] = L.PReLU(caffe_net['add{}'.format(layer_idx)])
    outblob = caffe_net['prelu{}'.format(layer_idx)]

    layer_idx += 1

    return outblob, layer_idx


def DownSampler(
                caffe_net,
                layer_idx,
                bottom_blob1,
                bottom_blob2, # always not none
                bottom_blob2_down_times,
                nin,
                nout,
                k=4,
                r_lim=9,
                reinf=True
                ):

    config_inp_reinf = 3
    nout_new = nout - nin

    # TODO: fix needed
    # caffe_net['pool{}'.format(layer_idx)] = L.Pooling(bottom_blob1, pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=2)
    caffe_net['pool{}'.format(layer_idx)] = L.Pooling(bottom_blob1, pool=P.Pooling.AVE, kernel_size=3, stride=2)

    avg_out = caffe_net['pool{}'.format(layer_idx)]
    layer_idx += 1
    eesp_out, layer_idx = EESP(caffe_net, layer_idx, bottom_blob1, nin, nout_new, stride=2, k=k, r_lim=r_lim, down_method='avg')
    caffe_net['cat{}'.format(layer_idx)] = L.Concat(*[eesp_out, avg_out], axis=1)
    output = caffe_net['cat{}'.format(layer_idx)]

    # need to know how many downsampling we need to do
    for i in range(bottom_blob2_down_times):
        # TODO: fix needed
        # caffe_net['pool{}'.format(layer_idx + i)] = L.Pooling(bottom_blob2, pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=2)
        caffe_net['pool{}'.format(layer_idx + i)] = L.Pooling(bottom_blob2, pool=P.Pooling.AVE, kernel_size=3, stride=2)

        bottom_blob2 = caffe_net['pool{}'.format(layer_idx + i)]
        layer_idx += 1

    bottom_blob, layer_idx = CBR(caffe_net, layer_idx, bottom_blob2, config_inp_reinf, 3, 1)
    bottom_blob, layer_idx = CB(caffe_net, layer_idx, bottom_blob, nout, 1, 1)
    caffe_net['add{}'.format(layer_idx)] = L.Eltwise(bottom_blob, output, operation=P.Eltwise.SUM, coeff=[1,1])
    output = caffe_net['add{}'.format(layer_idx)]
    layer_idx += 1
    caffe_net['prelu{}'.format(layer_idx)] = L.PReLU(output)

    return caffe_net['prelu{}'.format(layer_idx)], layer_idx + 1


def EfficientPWConv(
                    caffe_net,
                    layer_idx,
                    bottom_blob,
                    nin,
                    nout,
                    height,
                    width
                    ):

    groups = math.gcd(nin, nout)

    x, layer_idx = CBR(caffe_net, layer_idx, bottom_blob, nout, 3, 1, groups)




    # caffe_net['pool{}'.format(layer_idx)] = L.Pooling(bottom_blob,
    #                                                     pool=P.Pooling.AVE,
    #                                                     global_pooling=True)

    # caffe_net['conv{}'.format(layer_idx)] = L.Convolution(caffe_net['pool{}'.format(layer_idx)], 
    #                                                         num_output=nout,
    #                                                         kernel_size=1,
    #                                                         stride=1,
    #                                                         pad=0,
    #                                                         group=1,
    #                                                         bias_term=False)

    # caffe_net['sigmo{}'.format(layer_idx)] = L.Sigmoid(caffe_net['conv{}'.format(layer_idx)])

    '''
    caffe_net['tile{}'.format(layer_idx)] = L.Tile(caffe_net['sigmo{}'.format(layer_idx)], axis = 1, tiles = height*width)
    caffe_net['reshape{}'.format(layer_idx)] = L.Reshape(caffe_net['tile{}'.format(layer_idx)],
                                                                    reshape_param={'shape':{'dim': [0, nout, height, width]}})
    '''
    # caffe_net['flatten{}'.format(layer_idx)] = L.Flatten(caffe_net['sigmo{}'.format(layer_idx)])
    # caffe_net['scale{}'.format(layer_idx)] = L.Scale(*[x, caffe_net['flatten{}'.format(layer_idx)]], axis=0, bias_term=False)

    # wts = caffe_net['scale{}'.format(layer_idx)]

    # caffe_net['prod{}'.format(layer_idx)] = L.Eltwise(wts, x, operation=P.Eltwise.PROD)

    # return caffe_net['prod{}'.format(layer_idx)], layer_idx + 1
    return x, layer_idx


# def Shuffle(
#             caffe_net,
#             layer_idx,
#             bottom_blob,
#             groups
#             ):

def EfficientPyrPool(
                    caffe_net,
                    layer_idx,
                    bottom_blob,
                    in_planes,
                    proj_planes,
                    out_planes,
                    scales=[2.0, 1.0, 0.5, 0.1],
                    last_layer_br=True
                    ):
    scales.sort(reverse=True)
    x, layer_idx = CBR(caffe_net, layer_idx, bottom_blob, proj_planes, 1, 1)

    hs = []
    caffe_net['deconv{}'.format(layer_idx)] = L.Deconvolution(x,
                                                            param={"lr_mult": 1, "decay_mult": 1},
                                                            convolution_param=dict(
                                                                                num_output=proj_planes,
                                                                                bias_term=False,
                                                                                kernel_size=4,
                                                                                stride=2,
                                                                                pad=1,
                                                                                dilation=1,
                                                                                group=proj_planes
                                                                                )
                                                            )
    temp_bb = caffe_net['deconv{}'.format(layer_idx)]
    layer_idx += 1

    caffe_net['conv{}'.format(layer_idx)] = L.Convolution(temp_bb,
                                                        num_output=proj_planes,
                                                        kernel_size=3,
                                                        stride=1,
                                                        pad=1,
                                                        bias_term=False,
                                                        group=proj_planes)
    temp_bb = caffe_net['conv{}'.format(layer_idx)]
    layer_idx += 1

    caffe_net['pool{}'.format(layer_idx)] = L.Pooling(temp_bb, pool=P.Pooling.AVE, kernel_size=4, pad=1, stride=2)

    hs.append(caffe_net['pool{}'.format(layer_idx)])
    layer_idx += 1



    caffe_net['conv{}'.format(layer_idx)] = L.Convolution(x,
                                                        num_output=proj_planes,
                                                        kernel_size=3,
                                                        stride=1,
                                                        pad=1,
                                                        bias_term=False,
                                                        group=proj_planes)
    hs.append(caffe_net['conv{}'.format(layer_idx)])
    layer_idx += 1


    caffe_net['pool{}'.format(layer_idx)] = L.Pooling(x, pool=P.Pooling.AVE, kernel_size=4, pad=1, stride=2)
    temp_bb = caffe_net['pool{}'.format(layer_idx)]
    layer_idx += 1

    caffe_net['conv{}'.format(layer_idx)] = L.Convolution(temp_bb,
                                                        num_output=proj_planes,
                                                        kernel_size=3,
                                                        stride=1,
                                                        pad=1,
                                                        bias_term=False,
                                                        group=proj_planes)
    temp_bb = caffe_net['conv{}'.format(layer_idx)]
    layer_idx += 1

    caffe_net['deconv{}'.format(layer_idx)] = L.Deconvolution(temp_bb,
                                                            param={"lr_mult": 1, "decay_mult": 1},
                                                            convolution_param=dict(
                                                                                num_output=proj_planes,
                                                                                bias_term=False,
                                                                                kernel_size=4,
                                                                                stride=2,
                                                                                pad=1,
                                                                                dilation=1,
                                                                                group=proj_planes
                                                                                )
                                                            )
    hs.append(caffe_net['deconv{}'.format(layer_idx)])
    layer_idx += 1



    caffe_net['pool{}'.format(layer_idx)] = L.Pooling(x, pool=P.Pooling.AVE, kernel_size=12, pad=1, stride=10)
    temp_bb = caffe_net['pool{}'.format(layer_idx)]
    layer_idx += 1

    caffe_net['conv{}'.format(layer_idx)] = L.Convolution(temp_bb,
                                                        num_output=proj_planes,
                                                        kernel_size=3,
                                                        stride=1,
                                                        pad=1,
                                                        bias_term=False,
                                                        group=proj_planes)
    temp_bb = caffe_net['conv{}'.format(layer_idx)]
    layer_idx += 1

    caffe_net['deconv{}'.format(layer_idx)] = L.Deconvolution(temp_bb,
                                                            param={"lr_mult": 1, "decay_mult": 1},
                                                            convolution_param=dict(
                                                                                num_output=proj_planes,
                                                                                bias_term=False,
                                                                                kernel_size=12,
                                                                                stride=10,
                                                                                pad=1,
                                                                                dilation=1,
                                                                                group=proj_planes
                                                                                )
                                                            )
    hs.append(caffe_net['deconv{}'.format(layer_idx)])
    layer_idx += 1

    caffe_net['cat{}'.format(layer_idx)] = L.Concat(*hs, axis=1)
    out = caffe_net['cat{}'.format(layer_idx)]
    layer_idx += 1

    bottom_blob, layer_idx = BR(caffe_net, layer_idx, out)
    #TODO: I skip shuffle in merge_layer
    bottom_blob, layer_idx = CBR(caffe_net, layer_idx, bottom_blob, proj_planes, groups=proj_planes)

    caffe_net['conv{}'.format(layer_idx)] = L.Convolution(bottom_blob,
                                                        num_output=out_planes,
                                                        kernel_size=1,
                                                        stride=1,
                                                        bias_term=not last_layer_br)
    out = caffe_net['conv{}'.format(layer_idx)]
    layer_idx += 1
    if last_layer_br:
        bottom_blob, layer_idx = BR(caffe_net, layer_idx, out)
        return bottom_blob, layer_idx

    return out, layer_idx

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
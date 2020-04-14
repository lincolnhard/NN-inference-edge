import enum
import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np

class AttentionType(enum.IntEnum):
    CHANNEL = 1
    SPATIAL = 2

def SpatialAttentionLayer(caffe_net, layer_idx, bottom_blob, out_channel, bias_term=False, masks=None):
    names = ['conv{}a'.format(layer_idx),
             'conv{}b'.format(layer_idx),
             'sig{}'.format(layer_idx),
             'eltwise{}'.format(layer_idx)
            ]

    out_ch_size = out_channel

    if masks is not None:
        in_mask = masks[0]
        in_mask_ch_size = len(in_mask)  # original output channel size
        assert in_mask_ch_size == out_ch_size
        out_ch_size = int(np.sum(in_mask)) # pruned output channel size

    start_bottom_blob = bottom_blob

    caffe_net[names[0]] = L.Convolution(bottom_blob,
                                        num_output=1,
                                        bias_term=bias_term,
                                        pad=0,
                                        kernel_size=1,
                                        stride=1)

    caffe_net[names[1]] = L.Convolution(caffe_net[names[0]],
                                        num_output=out_ch_size,
                                        bias_term=bias_term,
                                        pad=0,
                                        kernel_size=1,
                                        stride=1)
    
    caffe_net[names[2]] = L.Sigmoid(caffe_net[names[1]])

    caffe_net[names[3]] = L.Eltwise(caffe_net[names[2]],
                                    start_bottom_blob,
                                    operation=P.Eltwise.PROD )
    
    return caffe_net[names[3]], layer_idx + 1


def SqeezeExcitationLayer(caffe_net, layer_idx, bottom_blob, in_channel, reduced_ch, height, width, bias_term=False):
    names = ['gPool{}'.format(layer_idx),
             'fc{}a'.format(layer_idx),
             'fc{}a_relu'.format(layer_idx),
             'fc{}b'.format(layer_idx),
             'fc{}b_sigmoid'.format(layer_idx),
             'tile{}'.format(layer_idx),
             'reshape{}'.format(layer_idx),
             'eltwise{}'.format(layer_idx),
            ]

    start_bottom_blob = bottom_blob

    caffe_net[names[0]] = L.Pooling(bottom_blob, pool=P.Pooling.AVE, global_pooling=True)
    caffe_net[names[1]] = L.InnerProduct(caffe_net[names[0]], num_output=reduced_ch, bias_term=bias_term)
    caffe_net[names[2]] = L.ReLU(caffe_net[names[1]], in_place=True)
    caffe_net[names[3]] = L.InnerProduct(caffe_net[names[2]], num_output=in_channel, bias_term=bias_term)
    caffe_net[names[4]] = L.Sigmoid(caffe_net[names[3]])
    
    caffe_net[names[5]] = L.Tile(caffe_net[names[4]], axis = 1, tiles = height*width)
    caffe_net[names[6]] = L.Reshape(caffe_net[names[5]], reshape_param={'shape':{'dim': [0, in_channel, height, width]}})
    caffe_net[names[7]] = L.Eltwise(caffe_net[names[6]], start_bottom_blob, operation=P.Eltwise.PROD )
    
    return caffe_net[names[7]], layer_idx + 1
    

# Just code test
if __name__ == '__main__':
    net = caffe.NetSpec()

    net['data'] = L.Input(shape=dict(dim=[1, 32, 360, 640]))
    net['output'], _ = SqeezeExcitationLayer(net, 0, net['data'], in_channel = 32, reduced_ch=4, height=360, width=640)
    print(str(net.to_proto()))
    print(net['data'])
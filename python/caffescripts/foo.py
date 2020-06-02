import cv2
import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np


net = caffe.NetSpec()
net['data'] = L.Input(shape=dict(dim=[1, 3, 6, 8]))
net['pool1'] = L.Pooling(net['data'], pool=P.Pooling.AVE, global_pooling=True)
net['sigmoid1'] = L.Sigmoid(net['pool1'])
# net['tile1'] = L.Tile(net['sigmoid1'], axis=1, tiles = 6*8) # not well supported
# net['reshape1'] = L.Reshape(net['tile1'], reshape_param={'shape':{'dim': [0, 3, 6, 8]}})

net['flatten1'] = L.Flatten(net['sigmoid1'])
net['scale1'] = L.Scale(*[net['data'], net['flatten1']], axis=0, bias_term=False)

net['eltwise1'] = L.Eltwise(net['scale1'], net['data'], operation=P.Eltwise.PROD )

with open('abc.prototxt', 'w') as f:
    f.write(str(net.to_proto()))

net = caffe.Net('abc.prototxt', caffe.TEST)
print(net.blobs['data'].data.shape)
# print(net.blobs['pool1'].data.shape)
# print(net.blobs['sigmoid1'].data.shape)
print(net.blobs['flatten1'].data.shape)
print(net.blobs['scale1'].data.shape)
print(net.blobs['eltwise1'].data.shape)



# for layername in list(net.params):
#         for i in range(len(net.params[layername])):
#                 net.params[layername][i].data[...] = np.array([random.randint(-100, 100)/100.0])


# net.blobs['data'].data[...] = np.random.rand(3, 6, 8)
# out = net.forward()

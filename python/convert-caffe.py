import numpy as np
from rknn.api import RKNN

CAFFE_TXT = '../data/espnetv2_segfcos.prototxt'
CAFFE_BIN = '../data/espnetv2_segfcos.caffemodel'
RKNN_MODEL = '../data/espnetv2_segfcos.rknn'

# Create RKNN object
rknn = RKNN()

# pre-process config
print('--> config model')
rknn.config(channel_mean_value='123.675 116.28 103.53 58.82', reorder_channel='0 1 2')
print('done')

# Load model
print('--> Loading model')
ret = rknn.load_caffe(model=CAFFE_TXT, proto='caffe', blobs=CAFFE_BIN)
if ret != 0:
    print('Load failed!')
    exit(ret)
print('done')

# Build model
print('--> Building model')
# ret = rknn.build(do_quantization=False)
ret = rknn.build(do_quantization=True, dataset='/home/toybrick/NN-inference-edge/pics640x480/dataset.txt')
if ret != 0:
    print('Build failed!')
    exit(ret)
print('done')

# Export rknn model
print('--> Export RKNN model')
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print('Export failed!')
    exit(ret)
print('done')


rknn.release()
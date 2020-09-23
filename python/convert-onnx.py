import os
import urllib
import traceback
import time
import sys
import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '../data/espnet_0716_det.onnx'
RKNN_MODEL = '../data/espnet_0716_det.rknn'

# Create RKNN object
rknn = RKNN(verbose=True)

# pre-process config
print('--> config model')
rknn.config(channel_mean_value='123.675 116.28 103.53 58.82', reorder_channel='0 1 2')
print('done')

# Load model
print('--> Loading model')
ret = rknn.load_onnx(model=ONNX_MODEL)
if ret != 0:
    print('Load failed!')
    exit(ret)
print('done')

# Build model
print('--> Building model')
ret = rknn.build(do_quantization=True, dataset='dataset.txt')
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
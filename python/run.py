from rknn.api import RKNN
import numpy as np

rknn = RKNN()

print('--> Config model')
rknn.config(channel_mean_value='0 0 0 255', reorder_channel='2 1 0')
print('done')

print('--> Loading model')
rknn.load_rknn('../data/nie-mobilenetssd-int8.rknn')
print('done')

print('--> Init runtime environment')
ret = rknn.init_runtime()
if ret != 0:
    print('Init runtime environment failed')
    exit(ret)
print('done')


img = np.random.randint(0, 255, size=(512, 512, 3), dtype=np.uint8) #HWC


print('--> Running model')
outputs = rknn.inference(inputs=[img])
print('done')

print('--> Begin evaluate model performance')
perf_results = rknn.eval_perf(inputs=[img])
print('done')

rknn.release()
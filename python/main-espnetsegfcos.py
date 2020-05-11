import os
os.environ['GLOG_minloglevel'] = '2' # suprress Caffe verbose prints
import torch
# import caffe
# from caffe import layers as L
# from caffe import params as P
import cv2
from torchvision import transforms
# import caffescripts
import pytorchscripts
from PIL import Image, ImageDraw
import numpy as np
# import torch.nn.functional as F


def cv2_draw(img, pts, line_color, line_width, is_closed=True, line_type=cv2.LINE_8, shift=0):
    pts = pts.numpy().astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=is_closed, color=pytorchscripts.get_bgr(line_color), 
                  thickness=line_width, lineType=line_type, shift=shift)

if __name__ == '__main__':

    #===================
    ### Mark groundtruth
    #===================
    baseroot = '/Users/lincolnlee/Documents/NN-inference-edge/data/'
    image_file = baseroot + 'VIRB0412_05002.jpg'
    # img_gt = cv2.imread(image_file) # shape: (height, width, channels)
    # json_content = pytorchscripts.load_json(image_file[:-3] + 'json')
    # for i in json_content['shapes']:
    #     points = np.array(i['points'], np.int32)
    #     cv2.polylines(img_gt, [points], True, (0, 0, 255), 2)
    # cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
    # cv2.imshow("demo", img_gt)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #======================
    ### Apply pytorch model
    #======================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = pytorchscripts.load_json('/Users/lincolnlee/Documents/gallopvision/espnet_0504/training.json')
    mean = model_config['dataset']['mean']
    std = model_config['dataset']['std']
    size = model_config['dataset']['size']
    labels = model_config['dataset']['labels']
    max_joints = model_config['model']['max_joints']
    joints_dim = model_config['model']['max_joints'] * 2
    strides=(8, 16, 32, 64, 128)
    num_classes = model_config['model']['num_classes_fcos'] - 1
    classes = model_config['dataset']['labels']
    joints = model_config['dataset']['joints']
    nms_pre = 1000

    model_config['model'].pop('name')
    pytorch_model = pytorchscripts.ESPNetV2Fusion(**model_config['model'])
    ckpt = torch.load('/Users/lincolnlee/Documents/gallopvision/espnet_0504/best.pth', map_location='cpu')
    pytorch_model.load_state_dict(ckpt['model'])
    pytorch_model.to(device)
    pytorch_model.eval()

    origimg = cv2.imread(image_file)
    img = cv2.resize(origimg, (size[1], size[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = (img[i][j] - mean) / std
    img = img.transpose((2, 0, 1))
    imgt = torch.from_numpy(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))


    with torch.no_grad():
        cls_scores, vertex_preds, centernesses, occlusions, bu_out = pytorch_model(imgt)

    score_thr = [0.4, 0.4]
    nms_thr = 0.5
    detection = pytorchscripts.FCOS_Detection(pytorch_model, labels, score_thr, nms_thr, size, joints)
    detected_shapes = detection(cls_scores, vertex_preds, centernesses, occlusions)

    dets = {}
    for d in detected_shapes:
        try:
            dets[d.label].append(d.score)
        except:
            dets[d.label] = [d.score]
    for i in dets:
        print(i, len(dets[i]), max(dets[i]))
    for s in detected_shapes:
        s.points[:,0] = s.points[:,0] * size[1]
        s.points[:,1] = s.points[:,1] * size[0]

    img_pred = cv2.imread(image_file)
    img_pred = cv2.resize(img_pred, (size[1], size[0]))
    for s in detected_shapes:
        cv2_draw(img_pred, s.points, 'red', 1)

    # cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
    cv2.imshow("demo", img_pred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    #=================
    ### Export to ONNX
    #=================

    # torch.onnx.export(pytorch_model,
    #                     imgt,
    #                     "espnetv2fusion.onnx",
    #                     export_params=True,
    #                     keep_initializers_as_inputs=True,
    #                     opset_version=10,
    #                     do_constant_folding=True,
    #                     input_names = ['input'],
    #                     output_names = ['cls_score', 'bbox_pred', 'centerness', 'occlusion', 'bu_out']
    #                 )

    #===================
    ### Apply ONNX model
    #===================

    # import onnx
    # onnx_model = onnx.load("espnetv2fusion.onnx")
    # onnx.checker.check_model(onnx_model)

    #==========================================================
    ### Create prototxt, caffemodel and feed with random weight
    #==========================================================

    fileName = 'espnetv2_segfcos'
    netW = model_config['dataset']['size'][1]
    netH = model_config['dataset']['size'][0]
    netClassesSeg = model_config['model']['num_classes_seg']
    netClassesFcos = model_config['model']['num_classes_fcos']
    net = caffe.NetSpec()
    net['data'] = L.Input(shape=dict(dim=[1, 3, netH, netW]))

    caffe_model = caffescripts.ESPNetV2Fusion()
    caffe_model.createPrototxt(net)

    with open(fileName + '.prototxt', 'w') as f:
        f.write(str(net.to_proto()))

    # fill random weight
    net = caffe.Net(fileName + '.prototxt', caffe.TEST)
    for layername in list(net.params):
            for i in range(len(net.params[layername])):
                    net.params[layername][i].data[...] = np.array([np.random.randint(-100, 100)/100.0])
    net.save(fileName + '.caffemodel')

    print(net.blobs['sigmo182'].data.shape, net.blobs['prelu183'].data.shape)
    # print(net.blobs['prelu254'].data.shape)

    #=======================================
    ### Feed weighting from pytorch to caffe
    #=======================================

    #=======
    ### Test
    #=======

    # net = caffe.Net(fileName + '.prototxt', fileName + '.caffemodel', caffe.TEST)
    # net.blobs['data'].data[...] = img
    # out = net.forward()

    # print(net.blobs['prelu254'].data.shape)
    # print(net.blobs['scoremap_perm'].data.shape)
    # print(net.blobs['centernessmap_perm'].data.shape)
    # print(net.blobs['occlusionmap_perm'].data.shape)
    # print(net.blobs['regressionmap_perm'].data.shape)
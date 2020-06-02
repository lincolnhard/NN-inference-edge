import os
os.environ['GLOG_minloglevel'] = '2' # suprress Caffe verbose prints
import torch
import caffe
from caffe import layers as L
from caffe import params as P
import cv2
from torchvision import transforms
import caffescripts
import pytorchscripts
from PIL import Image, ImageDraw
import numpy as np


def cv2_draw(img, pts, line_color, line_width, is_closed=True, line_type=cv2.LINE_8, shift=0):
    pts = pts.numpy().astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=is_closed, color=pytorchscripts.get_bgr(line_color), 
                  thickness=line_width, lineType=line_type, shift=shift)

if __name__ == '__main__':

    #===================
    ### Mark groundtruth
    #===================
    baseroot = '/Users/lincolnlee/Documents/NN-inference-edge/python/'
    image_file = baseroot + 'VIRB0695_07758.jpg'
 
    # img_gt = cv2.imread(image_file) # shape: (height, width, channels)
    # json_content = load_json(image_file[:-3] + 'json')
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
    model_config = pytorchscripts.load_json(baseroot + 'training.json')
    mean = model_config['dataset']['mean']
    std = model_config['dataset']['std']
    size = model_config['dataset']['size']
    labels = model_config['dataset']['labels']
    joints = model_config['dataset']['joints']

    model_config['model'].pop('name')
    pytorch_model = pytorchscripts.MnasNetA1FCOS(**model_config['model'])
    ckpt = torch.load(baseroot + 'best.pth', map_location='cpu')
    pytorch_model.load_state_dict(ckpt['model'])
    pytorch_model.to(device)
    pytorch_model.eval()

    '''
    img_pil = Image.open(image_file)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    imgt = transform(img_pil).unsqueeze(dim=0).to(device)
    '''

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
        cls_scores, vertex_preds, centernesses, occlusions = pytorch_model(imgt)

    
    score_thr = [0.5, 0.4, 0.4, 0.4, 0.4, 0.4]
    nms_thr = 0.5
    detection = pytorchscripts.FCOS_Detection(labels, score_thr, nms_thr, size, joints)
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
    

    #==========================================================
    ### Create prototxt, caffemodel and feed with random weight
    #==========================================================

    # fileName = 'mnasneta1fcos'
    # netW = model_config['dataset']['size'][1]
    # netH = model_config['dataset']['size'][0]
    # netClasses = model_config['model']['num_classes']
    # max_joints = model_config['model']['max_joints']
    # net = caffe.NetSpec()
    # net['data'] = L.Input(shape=dict(dim=[1, 3, netH, netW]))

    # caffe_model = caffescripts.MnasNetA1FCOS(num_classes=netClasses, width_mult=1.0, max_joints=max_joints)
    # caffe_model.createPrototxt(net, netH, netW)

    # with open(fileName + '.prototxt', 'w') as f:
    #     f.write(str(net.to_proto()))

    # # fill random weight
    # net = caffe.Net(fileName + '.prototxt', caffe.TEST)
    # for layername in list(net.params):
    #         for i in range(len(net.params[layername])):
    #                 net.params[layername][i].data[...] = np.array([np.random.randint(-100, 100)/100.0])
    # net.save(fileName + '.caffemodel')

    #=======================================
    ### Feed weighting from pytorch to caffe
    #=======================================

    # weighting_list = []
    # caffe_model.storeWeighting(weighting_list, pytorch_model)
    # caffescripts.feed_weighting(net, weighting_list)
    # net.save(fileName + '.caffemodel')




    #=======
    ### Test
    #=======

    # net = caffe.Net(fileName + '.prototxt', fileName + '.caffemodel', caffe.TEST)
    # net.blobs['data'].data[...] = img
    # out = net.forward()

    # cls_scores_caffe = torch.from_numpy(net.blobs['conv66'].data)
    # centernesses_caffe = torch.from_numpy(net.blobs['conv67'].data)
    # vertex_preds_caffe = torch.from_numpy(net.blobs['conv68'].data * 4.591407775878906)
    # occlusions_caffe = torch.from_numpy(net.blobs['conv69'].data)
    # print(torch.mean(torch.abs((cls_scores[0] - cls_scores_caffe))))
    # print(torch.mean(torch.abs((centernesses[0] - centernesses_caffe))))
    # print(torch.mean(torch.abs((vertex_preds[0] - vertex_preds_caffe))))
    # print(torch.mean(torch.abs((occlusions[0] - occlusions_caffe))))
    # print(torch.max(torch.abs((cls_scores[0] - cls_scores_caffe))))
    # print(torch.max(torch.abs((centernesses[0] - centernesses_caffe))))
    # print(torch.max(torch.abs((vertex_preds[0] - vertex_preds_caffe))))
    # print(torch.max(torch.abs((occlusions[0] - occlusions_caffe))))


    # detected_shapes = detection([cls_scores_caffe], [vertex_preds_caffe], [centernesses_caffe], [occlusions_caffe])

    # dets = {}
    # for d in detected_shapes:
    #     try:
    #         dets[d.label].append(d.score)
    #     except:
    #         dets[d.label] = [d.score]
    # for i in dets:
    #     print(i, len(dets[i]), max(dets[i]))
    # for s in detected_shapes:
    #     s.points[:,0] = s.points[:,0] * size[1]
    #     s.points[:,1] = s.points[:,1] * size[0]

    # img_pred = cv2.imread(image_file)
    # img_pred = cv2.resize(img_pred, (size[1], size[0]))
    # for s in detected_shapes:
    #     cv2_draw(img_pred, s.points, 'red', 1)

    # cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
    # cv2.imshow("demo", img_pred)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


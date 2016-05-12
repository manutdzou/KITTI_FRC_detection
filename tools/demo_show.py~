#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__','car','person','bike')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'vgg_m': ('VGG_CNN_M_1024',
                   'VGG_CNN_M_1024_faster_rcnn_final.caffemodel')}


def demo(net,image_list):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.ROOT_DIR, 'data', image_list[0])
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class

    ind=1
    color_list=[(255,0,0),(0,255,0),(0,0,255)]
    color_cls=[(0,255,255),(255,0,255),(255,255,0)]
    for j in range(1, len(CLASSES)):
        num_objs = int(image_list[ind+1])
        for i in xrange(num_objs):
            x1 = int(float(image_list[ind+2 + i * 4]))
            y1 = int(float(image_list[ind+3 + i * 4]))
            x2 = int(float(image_list[ind+4 + i * 4]))
            y2 = int(float(image_list[ind+5 + i * 4]))
            rect_start = (x1,y1)
            rect_end = (x2,y2)
            #cv2.rectangle(im, rect_start, rect_end, color_list[j-1], 2)
        ind+=4*num_objs+1

    thresh= 0.5
    NMS_THRESH = 0.3
    path = os.path.join(cfg.ROOT_DIR, 'data', 'results','show',image_list[0][17:])
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= thresh)[0]

        index=1
        if len(inds) == 0 and index==len(CLASSES[1:]):
            cv2.imwrite(path,im)
            return
        elif len(inds) == 0 and index<len(CLASSES[1:]):
            index+=1
            continue
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            x = bbox[0]
            y = bbox[1]
            rect_start = (x,y)
            x1 = bbox[2]
            y1 = bbox[3]
            rect_end = (x1,y1)
            color_pred=color_cls[cls_ind-1]
            cv2.rectangle(im, rect_start, rect_end, color_pred, 2)
    cv2.imwrite(path,im)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg_m')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)


    image_list=[]
    annotationfile = os.path.join(cfg.ROOT_DIR, 'data','KITTI_gt_val.txt')
    f = open(annotationfile)
    split_line = f.readline().strip().split()
    while(split_line):
        image_list.append(split_line)
        split_line = f.readline().strip().split()

    print '\n\nLoaded network {:s}'.format(caffemodel)


    for im_list in image_list:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        demo(net ,im_list)


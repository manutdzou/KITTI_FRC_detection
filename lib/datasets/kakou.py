#coding:utf-8
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.kakou
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class kakou(datasets.imdb):
    def __init__(self, image_set, devkit_path=None):
        datasets.imdb.__init__(self,image_set)#imageset 为train  test
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path)
        self._classes = ('__background__','car','person','bike')#包含的类
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))#构成字典{'__background__':'0','car':'1'}
        if self._image_set=='KakouTrain':
            self._image_index = self._load_image_set_index('KITTI_train_list.txt')#添加文件列表
        elif self._image_set=='KakouTest':
            self._image_index = self._load_image_set_index('KITTI_val_list.txt')#添加文件列表
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                   'use_salt' : True,
                   'top_k'    : 2000,
                   'use_diff' : False,
                   'rpn_file' : None}
        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
            return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):#根据_image_index获取图像路径
        image_path = os.path.join(self._data_path, index)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self, imagelist):#已经修改
        image_set_file = os.path.join(self._data_path, imagelist)# load ImageList that only contain ImageFileName
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):#若存在cache file则直接从cache file中读取
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self._load_annotation()  #已经修改，直接读入整个GT文件
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        
        return gt_roidb

    def selective_search_roidb(self):#已经修改
        cache_file = os.path.join(self.cache_path,self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file): #若存在cache_file则读取相对应的.pkl文件
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        if self._image_set !='KakouTest':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if self._image_set !='KakouTest':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):#已经修改
        #filename = os.path.abspath(os.path.join(self.cache_path, '..','selective_search_data',self.name + '.mat'))
        filename = os.path.join(self._data_path, 'regionboxes.mat')#这里输入相对应的预选框文件路径
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()
        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:,(1, 0, 3, 2)] - 1)#原来的Psacalvoc调换了列，我这里box的顺序是x1,y1,x2,y2 由EdgeBox格式为x1,y1,w,h经过修改    
            #box_list.append(raw_data[i][:,:] -1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_annotation(self):
        #，此函数作用读入GT文件，我的文件的格式 CarTrainingDataForFRCNN_1\Images\2015011100035366101A000131.jpg 1 147 65 443 361 
        gt_roidb = []
        annotationfile = os.path.join(self._data_path, 'KITTI_gt_train.txt')
        f = open(annotationfile)
        split_line = f.readline().strip().split()
        while(split_line):
            num_objs = int(split_line[1])
            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            ind=1
            num_ind=0
            for j in range(1, len(self._classes)):
                num=0
                for i in range(int(split_line[1+ind])):
                    x1 = float (split_line[ind+2 + i * 4])
                    y1 = float (split_line[ind+3 + i * 4])
                    x2 = float (split_line[ind+4 + i * 4])
                    y2 = float (split_line[ind+5 + i * 4])
                    cls =j
                    boxes[num_ind,:] = [x1, y1, x2, y2]
                    gt_classes[num_ind] = cls
                    overlaps[num_ind,cls] = 1.0
                    num+=1
                    num_ind+=1
                ind+=4*num+1


            overlaps = scipy.sparse.csr_matrix(overlaps)
            gt_roidb.append({'boxes' : boxes, 'gt_classes': gt_classes, 'gt_overlaps' : overlaps, 'flipped' : False})
            split_line = f.readline().strip().split()

        f.close()
        return gt_roidb

    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results')
        if os.path.isdir(path):
            pass
        else:
            os.mkdir(path)
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = path + '/' + comp_id + '_'+'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']
        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'detection_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir,'KITTI_val_list.txt',
                       'KITTI_gt_val.txt')
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.kakou('KakouTrain', '/home/bsl/KITTI_detection/data')
    res = d.roidb
    from IPython import embed; embed()

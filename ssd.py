#!/usr/bin/env python
# -*- coding: utf-8 -*-

CAFFE_ROOT = '/home/ubuntu/Codes/ssd'
LABEL_MAP = 'data/coco/labelmap_coco.prototxt'
# PROTO_TXT = 'models/VGGNet/coco/SSD_300x300/deploy.prototxt'
# CAFFE_MODEL = 'models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_240000.caffemodel'
# IMAGE_SIZE = 300
PROTO_TXT = 'models/VGGNet/coco/SSD_500x500/deploy.prototxt'
CAFFE_MODEL = 'models/VGGNet/coco/SSD_500x500/VGG_coco_SSD_500x500_iter_200000.caffemodel'
IMAGE_SIZE = 500
GPU = 0

import numpy as np
import os

import sys
sys.path.append(CAFFE_ROOT)
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))

import caffe
caffe.set_device(GPU)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2


class Detector(object):

    def __init__(self):
        # load MS COCO labels
        labelmap_file = os.path.join(CAFFE_ROOT, LABEL_MAP)
        file = open(labelmap_file, 'r')
        self._labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self._labelmap)

        model_def = os.path.join(CAFFE_ROOT, PROTO_TXT)
        model_weights = os.path.join(CAFFE_ROOT, CAFFE_MODEL)

        self._net = caffe.Net(model_def, model_weights, caffe.TEST)
        self._transformer = caffe.io.Transformer(
            {'data': self._net.blobs['data'].data.shape})
        self._transformer.set_transpose('data', (2, 0, 1))
        self._transformer.set_mean('data', np.array([104, 117, 123]))
        self._transformer.set_raw_scale('data', 255)
        self._transformer.set_channel_swap('data', (2, 1, 0))

        # set net to batch size of 1
        image_resize = IMAGE_SIZE
        self._net.blobs['data'].reshape(1, 3, image_resize, image_resize)


    def __call__(self, image):
        transformed_image = self._transformer.preprocess('data', image)
        self._net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self._net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = self._get_labelname(top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        # Make dictionary.
        objs = []
        for i in xrange(top_conf.shape[0]):
            info = {}
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            info['bbox'] = [xmin, ymin, xmax, ymax]
            info['label'] = top_labels[i]
            # info['score'] = top_conf[i]
            objs.append(info)

        return objs


    def _get_labelname(self, labels):
        num_labels = len(self._labelmap.item)
        labelnames = []
        if type(labels) is not list:
            labels = [labels]
        for label in labels:
            found = False
            for i in xrange(0, num_labels):
                if label == self._labelmap.item[i].label:
                    found = True
                    labelnames.append(self._labelmap.item[i].display_name)
                    break
            assert found == True
        return labelnames

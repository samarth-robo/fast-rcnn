import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np
import cv2
import os.path as osp
import time
import h5py
from IPython.core.debugger import Tracer

import sys
sys.path.insert(0, osp.expanduser('~/research/fast-rcnn/lib'))
from utils.blob import prep_im_for_blob
from fast_rcnn.config import cfg

sys.path.insert(0, osp.expanduser('~/research/dextro/dextro-tools/deep_context/python/utils'))
import vis

caffe.set_mode_gpu()
caffe.set_device(0)

year = 2007
dataset = 'trainval'
base_dir = osp.expanduser('~/research/VOCdevkit')
jpeg_image_path = osp.join(base_dir, 'VOC%d/JPEGImages'%year, '%s.jpg')
id_file = osp.join(base_dir, 'VOC%d/ImageSets/Main/%s.txt'%(year, dataset))
net = caffe.Net('../../models/VGG16/check_seg_fv.prototxt',
    osp.expanduser('~/research/dextro/dextro-tools/deep_context/snapshots/seg/_iter_30000.caffemodel'),
    caffe.TEST)
colors = np.load(osp.expanduser('~/research/dextro/dextro-tools/deep_context/data/seg_colors.npy'))

with open(id_file, 'r') as f:
  ids = [l.rstrip('\r').rstrip('\n') for l in f]
with h5py.File('../../data/cache/seg_fv_lmdbs/voc_%d_%s.h5' % (year, dataset), 'r') as f:
  fvdb = f['data']
  for i, im_id in enumerate(ids):
    print 'Image %d / %d' % (i, len(ids))
    start = time.time()
    im_filename = jpeg_image_path % im_id
    im = cv2.imread(im_filename)
    if im is None:
      print 'Could not read', im_filename
    processed_im, _ = prep_im_for_blob(im, cfg.PIXEL_MEANS,
        cfg.TRAIN.SCALES[0], cfg.TRAIN.MAX_SIZE)

    im_in = processed_im[np.newaxis, :, :, :].transpose((0, 3, 1, 2))

    net.blobs[net.inputs[0]].reshape(*(im_in.shape))
    net.blobs[net.inputs[0]].data[...] = im_in
    Tracer()()
    fv = np.asarray(fvdb[i][:-4], dtype=float)
    shape = np.asarray(fvdb[i][-4:], dtype=int)
    net.blobs[net.inputs[1]].reshape(*(shape))
    net.blobs[net.inputs[1]].data[...] = fv.reshape(shape)

    out = net.forward()
    probs = out[net.outputs[0]]
    probs = np.squeeze(probs).transpose((1, 2, 0)) 
    labels = np.argmax(probs, axis=2).astype(int)

    im_l = vis.create_labelled_image(labels, colors)
    im_l = cv2.resize(im_l, (im.shape[1], im.shape[0]))
    cv2.imshow('segmentation', im/2 + im_l/2)
    cv2.imshow('labels', im_l)
    cv2.imshow('image', im) 
    cv2.waitKey(-1)

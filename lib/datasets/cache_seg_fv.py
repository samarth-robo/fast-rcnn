import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np
import cv2
import lmdb
import os.path as osp
import time
from IPython.core.debugger import Tracer

import sys
sys.path.insert(0, osp.expanduser('~/research/fast-rcnn/lib'))
from utils.blob import prep_im_for_blob
from fast_rcnn.config import cfg

caffe.set_mode_gpu()
caffe.set_device(0)

year = 2007
dataset = 'trainval'
base_dir = osp.expanduser('~/research/VOCdevkit')
jpeg_image_path = osp.join(base_dir, 'VOC%d/JPEGImages'%year, '%s.jpg')
id_file = osp.join(base_dir, 'VOC%d/ImageSets/Main/%s.txt'%(year, dataset))
lmdb_path = '../../data/cache/seg_fv_lmdbs/voc_%s_%d' % (dataset, year)
db = lmdb.open(lmdb_path, map_size=int(1e12))
txn = db.begin(write=True)
net = caffe.Net('../../models/VGG16/cache_seg_fv_deploy.prototxt',
    '../../output/default/voc_2007_trainval/context_vgg16_1.0_iter_80000.caffemodel',
    caffe.TEST)

with open(id_file, 'r') as f:
  ids = [l.rstrip('\r').rstrip('\n') for l in f]

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
  out = net.forward()
  out = out[net.outputs[0]]

  dat = caffe.io.array_to_datum(out[0].astype(float), 0)
  txn.put('{:0>10d}'.format(i), dat.SerializeToString())
  print time.time() - start
  if i % 100 == 0:
    print 'Syncing LMDB'
    txn.commit()
    txn = lmdb.Transaction(env=db, write=True)

txn.commit()
db.close()

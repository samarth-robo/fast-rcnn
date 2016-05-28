import os, os.path as osp
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np
from IPython.core.debugger import Tracer

caffe.set_mode_gpu()
net = caffe.Net('models/VGG16/context_fv_deploy.prototxt', 
  'data/fast_rcnn_models/frcnn_coco_init.caffemodel', caffe.TEST)

def get_context_blob(im_blob):
  caffe.set_device(1)
  net.blobs[net.inputs[0]].reshape(*(im_blob.shape))
  net.blobs[net.inputs[0]].data[...] = im_blob
  out = net.forward()
  
  return out[net.outputs[0]]

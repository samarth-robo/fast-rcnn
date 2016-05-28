import os
# os.environ['GLOG_minloglevel'] = '2'
import numpy as np
import caffe
from IPython.core.debugger import Tracer

'''Tool to combine two nets, copy net 2 into net 1'''

def combine_nets(src_net, dst_net, src_layer_names, dst_layer_names):
  assert(len(src_layer_names) == len(dst_layer_names))
  for src_layer, dst_layer in zip(src_names, dst_names):
    for i in xrange(len(src_net.params[src_layer])):
      dst_net.params[dst_layer][i].data[...] = src_net.params[src_layer][i].data
    print 'Transferred', src_layer, '->', dst_layer

  return dst_net

proto1 = 'models/VGG16/test_context.prototxt'
model1 = 'output/default/voc_2007_trainval/context_batch_size_2_vgg16_1.0_iter_40000.caffemodel'
proto2 = 'models/VGG16/test_context.prototxt'
model2 = 'data/fast_rcnn_models/frcnn_coco_init.caffemodel'
save_model = 'output/default/voc_2007_trainval/context_batch_size_2_vgg16_1.0_combined.caffemodel'

layer_names_1 = ['conv3_1_seg', 'conv3_2_seg', 'conv3_3_seg', 'conv4_1_seg', 'conv4_2_seg', 'conv4_3_seg', 'conv5_1_seg', 'conv5_2_seg', 'conv5_3_seg', 'fc6_seg', 'fc7_seg']
layer_names_2 = ['conv3_1_seg', 'conv3_2_seg', 'conv3_3_seg', 'conv4_1_seg', 'conv4_2_seg', 'conv4_3_seg', 'conv5_1_seg', 'conv5_2_seg', 'conv5_3_seg', 'fc6_seg', 'fc7_seg']

net1 = caffe.Net(proto1, model1, caffe.TEST)
net2 = caffe.Net(proto2, model2, caffe.TEST)

save_net = combine_nets(net2, net1, layer_names_2, layer_names_1)
save_net.save(save_model)
print 'Surgery done, saved', save_model

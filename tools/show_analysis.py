import numpy as np
import _init_paths
import cPickle, pickle
import sys
import os.path as osp
import json
import skimage.io as io
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer

im_save_basepath = osp.expanduser('~/Pictures/eccv16_results/context_1.0/%s.png')
inc_dec = 'inc'

sys.path.insert(0, osp.expanduser('~/research/coco/PythonAPI'))
from pycocotools.coco import COCO
cocoGt = COCO('/media/TB/MSCOCO/annotations/instances_val2014.json')
image_idx = cocoGt.getImgIds()

base_json_path = '../output/coco_baseline/coco_2014_val/context_vgg16_1.0_iter_480000/analysis_%s_%s.json'
ctx_json_path = base_json_path % (inc_dec, 'ctx')
noctx_json_path = base_json_path % (inc_dec, 'noctx')
dataDir = '/media/TB/MSCOCO/'
ctx_json = json.load(open(ctx_json_path, 'r'))
noctx_json = json.load(open(noctx_json_path, 'r'))

cocoDt_ctx = cocoGt.loadRes(ctx_json_path)
cocoDt_noctx = cocoGt.loadRes(noctx_json_path)

scores = np.array([ann['score'] for ann in ctx_json])
idx = np.argsort(-scores)

for i in idx:
  print i
  imgId = ctx_json[i]['image_id'] 
  img = cocoGt.loadImgs(imgId)[0]
  im_filename = '%s/images/val2014/%s' % (dataDir, img['file_name'])
  I = io.imread(im_filename)
  category_id = ctx_json[i]['category_id']
  category_name = cocoGt.cats[category_id]['name']

  title = '%5.3g' % ctx_json[i]['score']
  plt.figure(); plt.imshow(I); plt.axis('off'); plt.title(title)
  annIds = cocoDt_ctx.getAnnIds(imgIds = imgId)
  anns = cocoDt_ctx.loadAnns(annIds)
  cocoDt_ctx.showAnns(anns)
  plt.draw()
  im_name = '%s_%s_ctx' % (category_name, inc_dec)
  plt.gcf().savefig(im_save_basepath%im_name, transparent=True, bbox_inches='tight', pad_inches=0)

  title = '%5.3g' % noctx_json[i]['score']
  plt.figure(); plt.imshow(I); plt.axis('off'); plt.title(title)
  annIds = cocoDt_noctx.getAnnIds(imgIds = imgId)
  anns = cocoDt_noctx.loadAnns(annIds)
  cocoDt_noctx.showAnns(anns)
  plt.draw()
  im_name = '%s_%s_noctx' % (category_name, inc_dec)
  plt.gcf().savefig(im_save_basepath%im_name, transparent=True, bbox_inches='tight', pad_inches=0)

plt.show()
plt.waitforbuttonpress()

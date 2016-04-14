import numpy as np
import _init_paths
import heapq
from utils.cython_nms import nms
import cPickle
import utils.cython_bbox as bbox_utils
import sys
import os.path as osp
import json
from IPython.core.debugger import Tracer

sys.path.insert(0, osp.expanduser('~/research/coco/PythonAPI'))
from pycocotools.coco import COCO
_COCO = COCO('/media/TB/MSCOCO/annotations/instances_val2014.json')
image_idx = _COCO.getImgIds()

DC_FILE = '../output/coco_baseline/coco_2014_val/context_vgg16_1.0_iter_480000/detections.pkl'
D_FILE = '../output/coco_baseline/coco_2014_val/baseline_vgg16_iter_480000/detections.pkl'
OUT_FILE = '../output/coco_baseline/coco_2014_val/context_vgg16_1.0_iter_480000/analysis_%s.json'

print 'Loading detections...'
D = cPickle.load(open(D_FILE, 'r'))
DC = cPickle.load(open(DC_FILE, 'r'))
print 'Done'

n_classes = len(D)
n_images = len(D[0])
assert(n_images == len(image_idx))
n_dets = 20
overlap_thresh = 0.5

cats = _COCO.loadCats(_COCO.getCatIds())
# classes = tuple(['__background__'] + [c['name'] for c in cats])
# ind_to_class = dict(zip(xrange(num_classes), classes))
# class_to_coco_cat_id = dict(zip([c['name'] for c in cats], _COCO.getCatIds()))
ind_to_coco_cat_id = dict(zip(xrange(1, n_classes), _COCO.getCatIds()))
Tracer()()

inc_heap = [[] for _ in xrange(n_classes)]
dec_heap = [[] for _ in xrange(n_classes)]

inc_vals = -float(np.inf) * np.ones(n_classes)
dec_vals = -float(np.inf) * np.ones(n_classes)
inc_dets = [{} for _ in xrange(n_classes)]
dec_dets = [{} for _ in xrange(n_classes)]

def match_detections(d, dc):
  M = np.zeros(d.shape[0])

  overlaps = bbox_utils.bbox_overlaps(d.astype(np.float), dc.astype(np.float))
  M = np.argmax(overlaps, axis=1)
  M_vals = np.max(overlaps, axis=1)
  M[M_vals < overlap_thresh] = -1

  return np.where(M >= 0)[0], M[M >= 0] 

for im_idx in xrange(n_images):
  image_id = image_idx[im_idx]
  for cls in xrange(1, n_classes):
    d = D[cls][im_idx]
    dc = DC[cls][im_idx]

    if d.shape[0] == 0 or dc.shape[0] == 0:
      continue

    # keep_d, keep_dc = match_detections(d[:, :4], dc[:, :4])
    # d = d[keep_d]
    # dc = dc[keep_dc]
    # if d.shape[0] == 0 or dc.shape[0] == 0:
    #   continue

    d = d[np.argmax(d[:, -1])][np.newaxis, :]
    dc = dc[np.argmax(dc[:, -1])][np.newaxis, :]
    ov = bbox_utils.bbox_overlaps(d[:, :4].astype(float), dc[:, :4].astype(float))
    if ov[0, 0] <= 0:
      continue

    print 'im %d cls %d' % (im_idx, cls)
    inc = dc[:, -1] - d[:, -1]
    dec = -inc.copy()

    max_inc_idx = np.argmax(inc)
    d_bbox = d[max_inc_idx, :4]
    d_bbox = np.hstack((d_bbox[0:2], d_bbox[2:4]-d_bbox[0:2]+1))
    d_score = d[max_inc_idx, 4]
    dc_bbox = dc[max_inc_idx, :4]
    dc_bbox = np.hstack((dc_bbox[0:2], dc_bbox[2:4]-dc_bbox[0:2]+1))
    dc_score = dc[max_inc_idx, 4]
    category_id = ind_to_coco_cat_id[cls]
    max_inc = {'d': {'image_id': image_id, 'category_id': category_id, 'bbox': d_bbox.tolist(), 'score': float(d_score)},
        'dc': {'image_id': image_id, 'category_id': category_id, 'bbox': dc_bbox.tolist(), 'score': float(dc_score)}}

    max_dec_idx = np.argmax(dec)
    d_bbox = d[max_dec_idx, :4]
    d_bbox = np.hstack((d_bbox[0:2], d_bbox[2:4]-d_bbox[0:2]+1))
    d_score = d[max_dec_idx, 4]
    dc_bbox = dc[max_dec_idx, :4]
    dc_bbox = np.hstack((dc_bbox[0:2], dc_bbox[2:4]-dc_bbox[0:2]+1))
    dc_score = dc[max_dec_idx, 4]
    category_id = ind_to_coco_cat_id[cls]
    max_dec = {'d': {'image_id': image_id, 'category_id': category_id, 'bbox': d_bbox.tolist(), 'score': float(d_score)},
        'dc': {'image_id': image_id, 'category_id': category_id, 'bbox': dc_bbox.tolist(), 'score': float(dc_score)}}

    if inc[max_inc_idx] > inc_vals[cls]:
      inc_vals[cls] = inc[max_inc_idx]
      inc_dets[cls] = max_inc
    if dec[max_dec_idx] > dec_vals[cls]:
      dec_vals[cls] = dec[max_dec_idx]
      dec_dets[cls] = max_dec

    # if len(inc_heap[cls]) < n_dets:
    #   heapq.heappush(inc_heap[cls], (-inc[max_inc_idx], max_inc))
    # else:
    #   heapq.heappushpop(inc_heap[cls], (-inc[max_inc_idx], max_inc))

    # if len(dec_heap[cls]) < n_dets:
    #   heapq.heappush(dec_heap[cls], (-dec[max_dec_idx], max_dec))
    # else:
    #   heapq.heappushpop(dec_heap[cls], (-dec[max_dec_idx], max_dec))

# cPickle.dump((inc_heap, dec_heap), open(OUT_FILE, 'w'))
# cPickle.dump((inc_dets, dec_dets), open(OUT_FILE, 'w'))

inc_ctx_json = []
inc_noctx_json = []
dec_ctx_json = []
dec_noctx_json = []

for cls in xrange(1, n_classes):
  if len(inc_dets[cls]) > 0:
    inc_ctx_json.append(inc_dets[cls]['dc'])
    inc_noctx_json.append(inc_dets[cls]['d'])
  if len(dec_dets[cls]) > 0:
    dec_ctx_json.append(dec_dets[cls]['dc'])
    dec_noctx_json.append(dec_dets[cls]['d'])

json.dump(inc_ctx_json, open(OUT_FILE%'inc_ctx', 'w'))
print OUT_FILE%'inc_ctx', 'saved'
json.dump(inc_noctx_json, open(OUT_FILE%'inc_noctx', 'w'))
print OUT_FILE%'inc_noctx', 'saved'
json.dump(dec_ctx_json, open(OUT_FILE%'dec_ctx', 'w'))
print OUT_FILE%'dec_ctx', 'saved'
json.dump(dec_noctx_json, open(OUT_FILE%'dec_noctx', 'w'))
print OUT_FILE%'dec_noctx', 'saved'

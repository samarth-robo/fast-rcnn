#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/coco_context_1.0_vgg16.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#   --snapshot snapshots/coco_vgg16_mcg/_iter_100000.solverstate \
time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/coco/solver_context.prototxt \
  --weights data/fast_rcnn_models/frcnn_coco_init.caffemodel \
  --cfg experiments/cfgs/coco.yml \
  --imdb coco_2014_minitrain \
  --iters 48000

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/coco/test_context.prototxt \
  --net output/coco_baseline/coco_2014_minitrain/context_vgg16_1.0_iter_48000.caffemodel \
  --cfg experiments/cfgs/coco.yml \
  --imdb coco_2014_minival \
  --num_dets 100 --comp

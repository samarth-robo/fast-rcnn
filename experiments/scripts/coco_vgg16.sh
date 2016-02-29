#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/coco_frcnn_mcg_vgg16_crowd.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# time ./tools/train_net.py --gpu $1 \
#   --solver models/VGG16/coco/solver.prototxt \
#   --weights data/imagenet_models/VGG16.v2.caffemodel \
#   --cfg experiments/cfgs/coco.yml \
#   --imdb coco_2014_train \
#   --iters 240000

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/coco/test.prototxt \
  --net output/coco_baseline/coco_2014_train/snapshots/coco_vgg16_mcg/vgg16_fast_rcnn_iter_240000.caffemodel \
  --cfg experiments/cfgs/coco.yml \
  --imdb coco_2015_test-dev \
  --num_dets 100 --comp

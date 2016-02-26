#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/coco_baseline_vgg16_crowd.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#  --weights ~/research/dextro/dextro-tools/deep_context/models/frcnn_coco_init.caffemodel \
# time ./tools/train_net.py --gpu $1 \
#   --solver models/VGG16/coco/solver.prototxt \
#   --snapshot snapshots/coco_vgg16_mcg/_iter_100000.solverstate \
#   --cfg experiments/cfgs/coco.yml \
#   --imdb coco_2014_train \
#   --iters 480000

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/coco/test_context.prototxt \
  --net output/coco_baseline/coco_2014_train/snapshots/coco_vgg16_mcg/_frcnn_mcg_iter_480000.caffemodel \
  --cfg experiments/cfgs/coco.yml \
  --imdb coco_2015_test-dev \
  --num_dets 100 --comp

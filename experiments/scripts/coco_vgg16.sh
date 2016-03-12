#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

# LOG="experiments/logs/coco_baseline_vgg16_crowd.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
LOG="experiments/logs/coco_baseline_vgg16.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# time ./tools/train_net.py --gpu $1 \
#   --solver models/VGG16/coco/solver.prototxt \
#   --weights data/imagenet_models/VGG16.v2.caffemodel \
#   --cfg experiments/cfgs/coco.yml \
#   --imdb coco_2014_train \
#   --iters 480000

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/coco/test.prototxt \
  --net output/coco_baseline/coco_2014_train/baseline_vgg16_iter_480000.caffemodel \
  --cfg experiments/cfgs/coco.yml \
  --imdb coco_2014_val \
  --num_dets 100 --comp

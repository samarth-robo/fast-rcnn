#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/coco_obj_context_0.25_vgg16.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# time ./tools/train_net.py --gpu $1 \
#   --solver models/VGG16/coco/solver_obj_context.prototxt \
#   --weights data/imagenet_models/VGG16.v2.caffemodel \
#   --cfg experiments/cfgs/coco.yml \
#   --imdb coco_2014_train \
#   --iters 480000

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/coco/test_obj_context.prototxt \
  --net output/coco_baseline/coco_2014_train/obj_context_vgg16_0.2_iter_480000.caffemodel \
  --cfg experiments/cfgs/coco.yml \
  --imdb coco_2015_test \
  --num_dets 100 --comp

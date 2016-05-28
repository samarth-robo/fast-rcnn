#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/coco_batch_size_2_context_1.0_vgg16.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#   --snapshot snapshots/coco_vgg16_mcg/_iter_100000.solverstate \
time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/coco/solver_context.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --cfg experiments/cfgs/coco.yml \
  --imdb coco_2014_train \
  --iters 240000

python combine_nets.py

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/coco/test_context.prototxt \
  --net output/coco_baseline/coco_2014_train/context_batch_size_2_vgg16_1.0_combined.caffemodel \
  --cfg experiments/cfgs/coco.yml \
  --imdb coco_2014_val \
  --num_dets 100 \
  --comp

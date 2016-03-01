#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/coco_baseline_vgg16_crowd.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#   --snapshot snapshots/coco_vgg16_mcg/_iter_100000.solverstate \
time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/coco/solver.prototxt \
  --weights ~/research/dextro/dextro-tools/deep_context/models/frcnn_coco_init.caffemodel \
  --cfg experiments/cfgs/coco.yml \
  --imdb coco_2014_minitrain \
  --iters 48000

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/coco/test_context.prototxt \
  --net output/coco_baseline/coco_2014_minitrain/_frcnn_mcg_iter_48000.caffemodel \
  --cfg experiments/cfgs/coco.yml \
  --imdb coco_2014_minival \
  --num_dets 100 --comp

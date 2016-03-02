#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/coco_obj_context_vgg16.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# time ./tools/train_net.py --gpu $1 \
#   --solver models/VGG16/coco/solver_obj_context.prototxt \
#   --weights data/imagenet_models/VGG16.v2.caffemodel \
#   --cfg experiments/cfgs/coco.yml \
#   --imdb coco_2014_minitrain \
#   --iters 48000

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/coco/test_obj_context.prototxt \
  --net output/coco_baseline/coco_2014_minitrain/obj_context_vgg16_iter_48000.caffemodel \
  --cfg experiments/cfgs/coco.yml \
  --imdb coco_2014_minival \
  --num_dets 100 --comp

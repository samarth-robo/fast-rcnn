#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

# LOG="experiments/logs/context_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
LOG="experiments/logs/context_0.25_vgg16.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu 0 \
  --solver models/VGG16/solver.prototxt \
  --weights data/fast_rcnn_models/frcnn_coco_init.caffemodel \
  --imdb voc_2007_trainval

time ./tools/test_net.py --gpu 0 \
  --def models/VGG16/test_context.prototxt \
  --net output/default/voc_2007_trainval/context_vgg16_0.2_iter_40000.caffemodel \
  --imdb voc_2007_test --comp

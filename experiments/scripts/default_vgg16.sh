#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

# LOG="experiments/logs/default_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
LOG="experiments/logs/baseline_vgg16.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb voc_2007_trainval \
  --iters 80000

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/test.prototxt \
  --net output/default/voc_2007_trainval/baseline_vgg16_iter_80000.caffemodel \
  --imdb voc_2007_test \
  --comp

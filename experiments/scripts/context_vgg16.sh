#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/context_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# time ./tools/train_net.py --gpu 0 \
#   --solver models/VGG16/solver.prototxt \
#   --weights ~/research/dextro/dextro-tools/deep_context/models/det_seg_frcnn.caffemodel \
#   --imdb voc_2007_trainval

time ./tools/test_net.py --gpu 0 \
  --def models/VGG16/test_context.prototxt \
  --net output/default/voc_2007_trainval/snapshots/vgg16/_iter_40000.caffemodel \
  --imdb voc_2007_test --comp

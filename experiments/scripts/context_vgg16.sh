#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

# LOG="experiments/logs/context_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
LOG="experiments/logs/context_1.0_vgg16.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu 0 \
  --solver models/VGG16/solver_context.prototxt \
  --weights data/fast_rcnn_models/frcnn_coco_init.caffemodel \
  --imdb voc_2007_trainval \
  --iters 40000

time ./tools/test_net.py --gpu 0 \
  --def models/VGG16/test_context.prototxt \
  --net output/default/voc_2007_trainval/context_batch_size_2_vgg16_1.0_iter_40000.caffemodel \
  --imdb voc_2007_test \
  --comp

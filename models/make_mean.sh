#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb

TOOLS=caffe/build/tools

$TOOLS/compute_image_mean train_lmdb \
  imagenet_mean.binaryproto

echo "Done."
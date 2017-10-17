#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/iam"))')
fi

python -u DeepHandwriting_ck.py \
  --train_files data/iam/iam-lower-train.csv \
  --dev_files data/iam/iam-lower-train.csv \
  --test_files data/iam/iam-lower-train.csv \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --n_hidden 494 \
  --epoch 50 \
  --checkpoint_dir "$checkpoint_dir" \
  "$@"

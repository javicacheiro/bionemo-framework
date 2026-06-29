#!/usr/bin/env bash
set -euo pipefail

# Recommended corpus for viral LoRA continue-pretraining that targets the
# eukaryotic-host virus gap in Evo2/OpenGenome2.
#
# Run from the root of this repository after making sure Evo2/Savanna's
# tools/preprocess_data.py is available at tools/preprocess_data.py.

python tools/preprocess_data.py \
  --input data/train.jsonl.gz \
  --output-prefix savanna/viral_euk_host_core_train \
  --tokenizer-type CharLevelTokenizer \
  --jsonl-keys text \
  --append-eod \
  --dataset-impl mmap

python tools/preprocess_data.py \
  --input data/valid.jsonl.gz \
  --output-prefix savanna/viral_euk_host_core_valid \
  --tokenizer-type CharLevelTokenizer \
  --jsonl-keys text \
  --append-eod \
  --dataset-impl mmap

#!/bin/bash

for lang in de en; do
  # CPU stages
  for backend in mllm bonsai bm-ensemble; do
    ANNIF_EVAL_JOBS=32 sbatch -p short -c 32 dvc-train-eval.sbatch $backend $lang
  done
  # GPU stages
  for backend in xtransformer bmx-ensemble; do
  ANNIF_EVAL_JOBS=1 sbatch -p gpu -G 1 -c 32 -t4:00:00 --mem=64G dvc-train-eval.sbatch $backend $lang
  done
done
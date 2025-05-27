#!/bin/bash

subset=$1  # dev or test
backend=$2 # e.g. bm-ensemble

for vocab in all tib-core; do
  for lang in de en de,en; do
    for origlang in de en; do

# for CPU jobs
      echo sbatch -p medium -c 2 generate-predictions.sbatch gnd-$vocab-$backend-LANG $lang \
        ../../submission/$subset-$vocab-$backend-$lang-$origlang.zip \
        ../../shared-task-datasets/TIBKAT/$vocab-subjects/data/$subset-$origlang.jsonl.gz

# for GPU jobs
      echo sbatch -p gpu -G 1 -c 2 generate-predictions.sbatch gnd-$vocab-$backend-LANG $lang \
        ../../submission/$subset-$vocab-$backend-$lang-$origlang.zip \
        ../../shared-task-datasets/TIBKAT/$vocab-subjects/data/$subset-$origlang.jsonl.gz
    done
  done
done

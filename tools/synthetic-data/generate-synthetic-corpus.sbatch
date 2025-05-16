#!/bin/bash
#SBATCH --job-name=synth-corpus
#SBATCH -o synth-corpus-output-%j.txt
#SBATCH -M ukko
#SBATCH -p gpu-oversub
#SBATCH --constraint=a100
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t4:00:00

# parameters
lang=$1
model=$2

echo "generating synthetic data for language $lang with model $model"

# setup Python environment and CUDA
module load Python cuDNN
source /wrk-vakka/group/natlibfi-annif/vllm-venv/bin/activate

# ensure vLLM is using only the Slurm-reserved GPU (not the shared one)
export CUDA_VISIBLE_DEVICES=0

# change to the correct directory
cd /wrk-vakka/group/natlibfi-annif/git/Annif-LLMs4Subjects-GermEval2025/tools/synthetic-data

# run the generation script
time python vllm-synthetic-data-gen.py \
  ../../shared-task-datasets/GND/GND-Subjects-all_dnb-skos-with-en-labels.ttl \
  $lang \
  $model \
  ../../data-sets/train-$lang.tsv.gz \
  ../../data-sets/new-synth-$lang-$model.tsv.gz

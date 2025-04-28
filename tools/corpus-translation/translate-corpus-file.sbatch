#!/bin/bash
#SBATCH --job-name=translate-corpus
#SBATCH -o translate-corpus-output-%j.txt
#SBATCH -M ukko
#SBATCH -p gpu-oversub
#SBATCH --constraint=a100
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t4:00:00

echo "Translating corpus file: $1 from language $2 using model $3"
echo -n "Number of records: "
unzip -l $1|tail -n 1|awk '{print $2}'

# activate Python & CUDA modules and vLLM virtual environment
module load Python cuDNN
source /wrk-vakka/group/natlibfi-annif/vllm-venv/bin/activate

# go to the right working directory
cd /wrk-vakka/group/natlibfi-annif/git/Annif-LLMs4Subjects-GermEval2025/tools/corpus-translation

# ensure vLLM is using only the Slurm-reserved GPU (not the shared one)
export CUDA_VISIBLE_DEVICES=0

# run the translation script
python vllm-dataset-to-parallel-corpus.py $1 $2 $3

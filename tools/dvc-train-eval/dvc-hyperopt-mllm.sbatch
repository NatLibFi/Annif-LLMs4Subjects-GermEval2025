#!/bin/bash
#SBATCH --job-name=dvc-hyperopt
#SBATCH -o dvc-hyperopt-output-%j.txt
#SBATCH -M ukko
#SBATCH --mem=64G
#SBATCH -t4:00:00
#SBATCH -c 64

lang=$1

echo "Params: $lang"

module purge
module load Python/3.11.5-GCCcore-13.2.0 CUDA

# activate python virtualenv with DVC and Annif
source /wrk-vakka/group/natlibfi-annif/git/Annif/venv-turso/bin/activate

# change to the correct directory
cd /wrk-vakka/group/natlibfi-annif/git/Annif-LLMs4Subjects-GermEval2025/

# run DVC hyperparameter optimization
dvc repro hyperopt-mllm@$lang

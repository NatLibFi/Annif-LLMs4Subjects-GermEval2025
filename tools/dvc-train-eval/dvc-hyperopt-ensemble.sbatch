#!/bin/bash
#SBATCH --job-name=dvc-hyperopt-ensemble
#SBATCH -o dvc-hyperopt-output-%j.txt
#SBATCH -M ukko
#SBATCH -p short
#SBATCH -c 32
#SBATCH --mem=20G
#SBATCH -t6:00:00

sources=$1
lang=$2

echo "DVC hyperopt ensemble. Params: $sources $lang"

module purge
module load Python/3.11.5-GCCcore-13.2.0 CUDA

# activate python virtualenv with DVC and Annif
source /wrk-vakka/group/natlibfi-annif/git/Annif/venv-turso/bin/activate

# change to the correct directory
cd /wrk-vakka/group/natlibfi-annif/git/Annif-LLMs4Subjects-GermEval2025/

# we can run up to 8 eval jobs
export ANNIF_EVAL_JOBS=8

# 4 secs sleep between DVC starts to try to avoid DVC lock contention
# see https://github.com/iterative/dvc/issues/755#issuecomment-1083079350
SLEEP_DURATION=$(( (SLURM_JOB_ID % 16) * 4 ))
echo "Sleeping $SLEEP_DURATION seconds..."
sleep $SLEEP_DURATION

# run DVC to train and evaluate models
dvc repro hyperopt-ensemble@$sources-$lang

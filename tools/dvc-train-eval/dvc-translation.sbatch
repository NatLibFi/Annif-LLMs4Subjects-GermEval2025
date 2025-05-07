#!/bin/bash
#SBATCH --job-name=dvc-translation
#SBATCH -o dvc-translation-output-%j.txt
#SBATCH -M ukko
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH -t0:30:00

model=$1
backend=$2
lang=$3

echo "DVC translation. Params: $model $backend $lang"

module purge
module load Python/3.11.5-GCCcore-13.2.0

# activate python virtualenv with DVC and Annif
source /wrk-vakka/group/natlibfi-annif/git/Annif/venv-turso/bin/activate

# change to the correct directory
cd /wrk-vakka/group/natlibfi-annif/git/Annif-LLMs4Subjects-GermEval2025/

# we can run up to 8 eval jobs
export ANNIF_EVAL_JOBS=8

# 4 secs sleep between DVC starts to try to avoid DVC lock contention
# see https://github.com/iterative/dvc/issues/755#issuecomment-1083079350
SLEEP_DURATION=$(( (SLURM_JOB_ID % 32) * 4 ))
echo "Sleeping $SLEEP_DURATION seconds..."
sleep $SLEEP_DURATION

# run DVC to train and evaluate models
dvc --cd translation/$model repro eval@$backend-$lang

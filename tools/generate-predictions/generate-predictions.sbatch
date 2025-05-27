#!/bin/bash
#SBATCH --job-name=generate-predictions
#SBATCH -o generate-predictions-output-%j.txt
#SBATCH -M ukko
#SBATCH --mem=16G
#SBATCH -t12:00:00

echo "Params: $@"

module purge
module load Python/3.11.5-GCCcore-13.2.0 CUDA

# activate python virtualenv with DVC and Annif
source /wrk-vakka/group/natlibfi-annif/git/Annif/venv-turso/bin/activate

bash generate-predictions.sh $@

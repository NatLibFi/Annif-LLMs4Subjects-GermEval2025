#!/bin/bash
#SBATCH --job-name=translate-corpus
#SBATCH -o translate-corpus-output-%j.txt
#SBATCH -M ukko
#SBATCH -p gpu
#SBATCH --constraint=a100
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t20:00:00

echo "Translating corpus file: $1"
echo -n "Number of records: "
wc -l $1
 
# aktivoidaan oikeanlainen python-ympäristö
module load CUDA
source /wrk-vakka/group/natlibfi-annif/vllm-cuda-venv/bin/activate

# mennään oikeaan paikkaan mistä tiedostot löytyvät
cd /wrk-vakka/group/natlibfi-annif/git/Annif-LLMs4Subjects/tools/corpus-translation
 
# ajetaan skripti
python vllm-datasets-to-parallel-corpus.py $1 

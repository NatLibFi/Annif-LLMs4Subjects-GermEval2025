#!/bin/bash
#SBATCH --job-name=generate-predictions-vllm
#SBATCH -o generate-predictions-vllm-output-%j.txt
#SBATCH -M ukko
#SBATCH -p gpu
#SBATCH --constraint=a100
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t24:00:00

# Command line parameters:
# 1.  vLLM model ID
# 2.  Annif project ID, e.g. "gnd-mllm-de"
# 3.  Language: can be "de" or "en"
# 4.  Path where predictions are written as a JSONL file
# 5+. Path(s) to zipped JSONL file(s) with input documents

vllm_model=$1
shift

echo "vLLM model: $vllm_model"
echo "Generation params: $@"

# Function to start vLLM service in the background
start_vllm() {
    echo "starting vllm with model $vllm_model"
    # Start vllm in the background and redirect output to a log file
    vllm serve $vllm_model --port $VLLM_PORT > vllm-output-$SLURM_JOB_ID.txt 2>&1 &
}

# Function to check if vLLM API is available
check_vllm_api() {
    until $(curl --output /dev/null --silent --fail http://localhost:$VLLM_PORT/v1/models); do
        echo "Waiting for vLLM API to become available..."
        sleep 2
    done
    echo "vLLM API is available"
}

# Function to start Annif service in the background
start_annif() {
    annif run --port $ANNIF_PORT &
    ANNIF_PID=$!
    echo "Annif service started on port $PORT with PID $ANNIF_PID"
    # change back to the old directory
    cd -
}

# Function to check if Annif REST API is available
check_annif_api() {
    until $(curl --output /dev/null --silent --head --fail http://127.0.0.1:$ANNIF_PORT/v1/); do
        echo "Waiting for Annif REST API to become available on port $PORT..."
        sleep 2
    done
    echo "Annif REST API is available on port $PORT."
}

# Function to stop Annif service
stop_annif() {
    if [ -n "$ANNIF_PID" ]; then
        kill $ANNIF_PID
        echo "Annif service stopped."
    fi
}

# Extract the last three digits of SLURM_JOB_ID and calculate the port numbers
ANNIF_PORT=$((12000 + ${SLURM_JOB_ID: -3}))
VLLM_PORT=$((13000 + ${SLURM_JOB_ID: -3}))

# Activate Python virtual env for vLLM
module purge
module load Python/3.12.3-GCCcore-13.3.0 CUDA
source /wrk-vakka/group/natlibfi-annif/vllm-venv/bin/activate

# ensure vLLM is using only the Slurm-reserved GPU (not the shared one)
export CUDA_VISIBLE_DEVICES=0

# Start vLLM service
start_vllm

# Wait for vLLM REST API to become available
check_vllm_api

# Activate python virtualenv with DVC and Annif
module purge
module load Python/3.11.5-GCCcore-13.2.0 CUDA
source /wrk-vakka/group/natlibfi-annif/git/Annif/venv-turso/bin/activate

# Configure Annif llm backend to use the vLLM API
export LLM_API_BASE_URL=http://localhost:$VLLM_PORT/v1

# change to the correct directory for starting Annif
cd /wrk-vakka/group/natlibfi-annif/git/Annif-LLMs4Subjects-GermEval2025/

# Start Annif service
start_annif

# Wait for Annif REST API to become available
check_annif_api

# Generate predictions by accessing the Annif REST API
python3 generate-predictions.py $ANNIF_PORT $@

# Stop Annif service
stop_annif

exit 0

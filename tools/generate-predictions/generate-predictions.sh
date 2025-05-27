#!/bin/bash

# Command line parameters:
# 1.  Annif project ID, where LANG stands for language ID, e.g. "gnd-all-mllm-LANG"
# 2.  Language: can be "de", "en" or "de,en"
# 3.  Path to zip file where predictions are written as JSON files
# 4+. Path(s) to zipped JSONL file(s) with input documents


# Function to start Annif service in the background
start_annif() {
    # change to the correct directory for starting Annif
    cd /wrk-vakka/group/natlibfi-annif/git/Annif-LLMs4Subjects/
    annif run --port $PORT &
    ANNIF_PID=$!
    echo "Annif service started on port $PORT with PID $ANNIF_PID"
    # change back to the old directory
    cd -
}

# Function to check if Annif REST API is available
check_annif_api() {
    until $(curl --output /dev/null --silent --head --fail http://127.0.0.1:$PORT/v1/); do
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

# Extract the last three digits of SLURM_JOB_ID and calculate the port number
PORT=$((12000 + ${SLURM_JOB_ID: -3}))

# Start Annif service
start_annif

# Wait for Annif REST API to become available
check_annif_api

# Generate predictions by accessing the REST API
python3 generate-predictions.py $PORT $@

# Stop Annif service
stop_annif

exit 0


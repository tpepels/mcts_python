#!/bin/bash

# Activate the virtual environment
source $HOME/.mcts_venv/bin/activate

# The directory to scan for JSON files
DIRECTORY="$1"

# Flag to check if any JSON files are found
json_found=false

# Iterate over .json files and run each experiment
for json_file in "$DIRECTORY"/*.json; do
    if [ -e "$json_file" ]; then  # Check if json files exist in the directory
        echo "Running experiment for $json_file"
        BASENAME=$(basename -- "$json_file")
        UNIQUE_NAME="${BASENAME%.*}_$$"  # Remove extension and append PID

        # Construct output and error file paths with the identifier
        OUT_FILE="/home/tpepels/out/${UNIQUE_NAME}.out"
        ERR_FILE="/home/tpepels/out/err/${UNIQUE_NAME}.err"

        # Run the experiment using the Python script
        python -O experiments.py -b $HOME/results -j "$json_file" -c > "$OUT_FILE" 2> "$ERR_FILE"
        json_found=true
    fi
done

# Notify if no JSON files were found
if [ "$json_found" = false ]; then
    echo "No JSON files found in the directory."
    exit 1
fi

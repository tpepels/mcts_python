#!/bin/bash

# Check if the minimum number of arguments (directory path) is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_directory>"
    exit 1
fi

# Remove trailing slash if present from directory path
DIRECTORY="${1%/}"

# Construct the job script name
JOB_SCRIPT="scripts/job_script.sh"

# Check if the job script exists
if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Job script $JOB_SCRIPT does not exist."
    exit 1
fi

# Initialize a variable to track if any JSON files are found
json_found=false

# Iterate over .json files and run each experiment sequentially
for json_file in "$DIRECTORY"/*.json; do
    if [ -e "$json_file" ]; then  # Check if json files exist in the directory
        echo "Running experiment for $json_file"
        BASENAME=$(basename -- "$json_file")
        
        # Construct output and error file paths with the identifier
        OUT_FILE="/home/tpepels/out/${BASENAME}.out"
        ERR_FILE="/home/tpepels/out/err/${BASENAME}.err"

        # Run the experiment using the job script
        bash "$JOB_SCRIPT" "$json_file" > "$OUT_FILE" 2> "$ERR_FILE"
        json_found=true
    fi
done

# If no JSON files were found, notify the user
if [ "$json_found" = false ]; then
    echo "No JSON files found in the directory."
    exit 1
fi

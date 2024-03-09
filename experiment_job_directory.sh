#!/bin/bash

# Check if the directory path parameter is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_directory>"
    exit 1
fi

# Remove trailing slash if present
DIRECTORY="${1%/}"

# Iterate over .json files and submit each as a separate job
for json_file in "$DIRECTORY"/*.json; do
    if [ -e "$json_file" ]; then  # Check if json files exist in the directory
        echo "Submitting job for $json_file"
        sbatch run_experiment.sh "$json_file"
    else
        echo "No JSON files found in the directory."
        exit 1
    fi
done

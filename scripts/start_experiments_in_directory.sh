#!/bin/bash

# Check if the minimum number of arguments (directory path, city, duration) is provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <path_to_directory> <city> <duration>"
    echo "Cities: rome, genoa, ..."
    echo "Durations: 6h, 12h, 1d, 2d"
    exit 1
fi

# Remove trailing slash if present from directory path
DIRECTORY="${1%/}"

# City and Duration parameters
CITY="$2"
DURATION="$3"

# Set the number of CPUs based on the city
case $CITY in
    genoa)
        CPUS=192
        ;;
    rome)
        CPUS=128
        ;;
    *)
        echo "Invalid city. Valid options are: genoa, rome."
        exit 1
        ;;
esac

# Convert duration to sbatch compatible format
case $DURATION in
    6h)
        SBATCH_DURATION="6:00:00"
        ;;
    12h)
        SBATCH_DURATION="12:00:00"
        ;;
    1d)
        SBATCH_DURATION="24:00:00"
        ;;
    2d)
        SBATCH_DURATION="48:00:00"
        ;;
    *)
        echo "Invalid duration format."
        exit 1
        ;;
esac

# Construct the job script name based on city and duration
JOB_SCRIPT="scripts/slurm_job_script.sh"

# Check if the job script exists
if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Job script $JOB_SCRIPT does not exist."
    exit 1
fi

# Initialize a variable to track if any JSON files are found
json_found=false

# Iterate over .json files and submit each as a separate job
for json_file in "$DIRECTORY"/*.json; do
    if [ -e "$json_file" ]; then  # Check if json files exist in the directory
        echo "Submitting job for $json_file using $JOB_SCRIPT on partition $CITY with duration $SBATCH_DURATION using $CPUS CPUs per task"
        BASENAME=$(basename -- "$json_file")
        
        # Construct output and error file paths with the identifier
        OUT_FILE="/home/tpepels/out/${BASENAME}_%j.out"
        ERR_FILE="/home/tpepels/out/err/${BASENAME}_%j.err"

        sbatch -p "$CITY" --time="$SBATCH_DURATION" --cpus-per-task=$CPUS --exclusive --job-name="$BASENAME" -o "$OUT_FILE" -e "$ERR_FILE" "$JOB_SCRIPT" "$json_file"
        json_found=true
    fi
done

# If no JSON files were found, notify the user
if [ "$json_found" = false ]; then
    echo "No JSON files found in the directory."
    exit 1
fi

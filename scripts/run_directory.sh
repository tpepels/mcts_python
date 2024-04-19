#!/bin/bash

# Check if the minimum number of arguments (directory path) is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_directory>"
    exit 1
fi

# Remove trailing slash if present from directory path
DIRECTORY="${1%/}"

# Generate a unique screen name based on timestamp
SCREEN_NAME="experiment_$(date +%Y%m%d_%H%M%S)"

# Start a detached screen session to run the experiments
screen -dmS "$SCREEN_NAME" /bin/bash -c "scripts/run_experiments.sh \"$DIRECTORY\"; exec sh"

# Notify the user where to find the running experiments
echo "All experiments are running in screen session named: $SCREEN_NAME"
echo "Use 'screen -r $SCREEN_NAME' to attach to the session."
echo "Use 'screen -S $SCREEN_NAME -X quit' to kill the session."

# Add temporary aliases
echo "Adding temporary aliases..."
alias attach_experiment="screen -r $SCREEN_NAME"
alias kill_experiment="screen -S $SCREEN_NAME -X quit"

echo "Use 'attach_experiment' to attach to the session."
echo "Use 'kill_experiment' to kill the session."

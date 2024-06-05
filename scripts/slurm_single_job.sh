#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tom.pepels@maastrichtuniversity.nl

# Check if the time limit parameter is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_json_file>"
    exit 1
fi

#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source $HOME/.mcts_venv/bin/activate

# Navigate to the directory containing your Python program
cd $HOME/mcts_python

# Use the provided file parameter in the srun command
srun python -O experiments.py -b $HOME/results -j $1 -c

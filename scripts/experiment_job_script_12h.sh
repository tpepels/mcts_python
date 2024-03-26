#!/bin/bash
#SBATCH -p genoa
#SBATCH -J mcts
#SBATCH -o /home/tpepels/out/%J.out
#SBATCH -e /home/tpepels/err/%J.err
#SBATCH -t 720
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
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

# Navigate to the directory containing your Python program
cd $HOME/mcts_python

# Use the provided file parameter in the srun command
srun python -O experiments.py -n 192 -b $HOME/results -j $1 -c

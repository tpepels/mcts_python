#!/bin/bash
#SBATCH -p genoa
#SBATCH -J MCTSExperiments
#SBATCH -o /home/tpepels/out/%J.out
#SBATCH -e /home/tpepels/out/err_%J.err
#SBATCH -t 3000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive

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
srun python experiments.py -n 192 -b $HOME -j $1 -c

#!/bin/bash
#SBATCH -p genoa
#SBATCH -J GeneticEvaluationFunction
#SBATCH -o /home/tpepels/Out/Out_%J.out
#SBATCH -e /home/tpepels/Out/Err_%J.err
#SBATCH -t 600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.pepels@maastrichtuniversity.nl

#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Navigate to the directory containing your Python program
cd $HOME/mcts_python

srun python genetic_optimization.py --n_procs 192 --base_path $HOME eval_optimise.json
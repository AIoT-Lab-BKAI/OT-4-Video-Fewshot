#!/bin/bash
#SBATCH --job-name=c3d
#SBATCH --output=c3d_output.txt
#SBATCH --error=c3d_error.txt

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:1
#SBATCH --nodelist=slurmnode3
source activate $1
# Load any required modules
conda activate ot1
# Your commands or executable here
python trainers/C3DTrainer.py
srun --nodelist=slurmnode2 --pty bash  -i
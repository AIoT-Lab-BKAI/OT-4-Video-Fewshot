#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1          
#SBATCH --gres=gpu:1
#SBATCH --nodelist=slurmnode2

#SBATCH --mem=0
#SBATCH --job-name=c3d
#SBATCH --output=job_output/c3d_output.txt
#SBATCH --error=job_output/c3d_error.txt
source /home/admin/miniconda3/bin/activate
# activate conda env
conda activate ot1
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above

srun python trainers/C3DTrainer.py
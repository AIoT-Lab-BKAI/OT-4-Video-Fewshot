#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodelist=slurmnode5,slurmnode6
#SBATCH --nodes=2          
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=0
#SBATCH --job-name=trx
#SBATCH --output=trx_output.txt
#SBATCH --error=trx_error.txt
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
srun python trainers/TRXTrainer.py
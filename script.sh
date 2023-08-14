# this is main command, dont use is for debugging
python train.py config/opt.yaml
# this is the debug command
python train.py config/opt_test.yaml
srun --nodelist=slurmnode4 --pty bash -i
python trainer.py
python trainers/C3DTrainer.py
srun --nodelist=slurmnode3 --pty bash -i
srun --nodelist=slurmnode3 nvidia-smi

deepspeed --include localhost:1,2 train.py --deepspeed --deepspeed_config configs/ds_config.json
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  python trainers/TRXTrainer.py
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  python trainers/TRXTrainer.py --cfg configs/m1.py
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  python runs/train_molo_ot.py
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  python runs/train_molo_ot.py --log



deepspeed deep_speed/train.py --deepspeed --deepspeed_config deep_speed/config.json
# this is main command, dont use is for debugging
python train.py config/opt.yaml
# this is the debug command
python train.py config/opt_test.yaml
srun --nodelist=slurmnode4 --pty bash -i
python trainer.py
python trainers/C3DTrainer.py
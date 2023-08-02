import numpy as np
import scipy
from torch.utils.data import DataLoader
from easyfsl.samplers import TaskSampler
import os
import pytorch_lightning as pl
import yaml
import torch
import datetime


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

def get_episodic_dataloader(dataset, n_way, n_shot, n_query, n_tasks, num_workers=4):
    sampler = TaskSampler(dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=num_workers, pin_memory=True, collate_fn=sampler.episodic_collate_fn)
    return dataloader

def get_wandb_logger(logger_cfg):
    with open('secrets.yaml', 'r') as f:
        secrets = yaml.safe_load(f)
    os.environ["WANDB_API_KEY"] = secrets['wandb_api_key']
    logger = pl.loggers.WandbLogger(**logger_cfg)
    return logger

def get_checkpoint_callback(ckpt_cfg):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        **ckpt_cfg
    )
    return checkpoint_callback

def accuracy(logits, labels, calc_mean=True):
    correct = torch.argmax(logits, 1) == labels
    if calc_mean:
        return torch.mean(correct.float())
    else:
        return correct.float()
    
def get_workdir(root_dir):
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    # Convert the current date and time to a string
    formatted_datetime = current_datetime.strftime('%m_%d_%H_%M')
    # Print the formatted datetime string
    path = os.path.join(root_dir, formatted_datetime)
    os.makedirs(path, exist_ok=True)
    return path
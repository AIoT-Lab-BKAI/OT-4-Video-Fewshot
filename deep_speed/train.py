from argparse import ArgumentParser
from datasets.UCF_TRX import UCF_TRX
from datasets.UCF_model1 import FS_DS
from models.TRX import CNN_TRX
from torch.utils.data import DataLoader
import configs.TRX as cfg
from deepspeed.pipe import PipelineModule
import deepspeed
import torch

def get_dataloader():
    dataloader = {}
    for name, ds_cfg in cfg.dataset.items():
        dataset = UCF_TRX(**ds_cfg)
        dataset = FS_DS(dataset, **cfg.sampler[name])
        dataloader[name] = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    return dataloader

def get_dataset():
    dataset = UCF_TRX(**cfg.dataset['train'])
    dataset = FS_DS(dataset, **cfg.sampler['train'])
    return dataset

def get_model():
    model = CNN_TRX(cfg.model)
    return model

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    dataset = get_dataset()
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    engine, optimizer, loader, _ = deepspeed.initialize(
        args = args,
        model = model,
        training_data = dataset)
    
    model = PipelineModule(model, num_stages=1)    
    
    for batch in loader:
        print(batch[0].shape)



    
    
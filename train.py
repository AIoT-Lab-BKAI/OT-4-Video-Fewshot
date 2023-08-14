from basic_module import BasicModule
from datasets.UCF_TRX import UCF_TRX
from datasets.UCF_model1 import FS_DS
from models.M1 import M1
import pytorch_lightning as pl
import configs.m1 as cfg
from torch.utils.data import DataLoader
def get_dataloader():
    dataloader = {}
    for name, ds_cfg in cfg.dataset.items():
        dataset = UCF_TRX(**ds_cfg)
        dataset = FS_DS(dataset, **cfg.sampler[name])
        dataloader[name] = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    return dataloader

def train():
    model = M1(**cfg.model)
    pl_module = BasicModule(model)
    trainer = pl.Trainer(**cfg.trainer)
    loader = get_dataloader()
    trainer.fit(pl_module, loader['train'], loader['val'])
    
if __name__ == '__main__':
    train()
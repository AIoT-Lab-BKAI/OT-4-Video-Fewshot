import json
import os
from datasets.UCF_TRX import UCF_TRX
from models.TRX import CNN_TRX
from torch.optim.lr_scheduler import MultiStepLR
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils.utils import get_episodic_dataloader, get_workdir, mean_confidence_interval, accuracy
import yaml
import torch.nn.functional as F
import configs.TRX as cfg
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

def criterion(pred_logits, target_labels):
    res = {}
    res['loss'] = F.cross_entropy(pred_logits, target_labels)
    res['acc'] = accuracy(pred_logits, target_labels)
    return res

class TRXModel(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        
        self.cfg = cfg
    

    def training_step(self, batch, batch_idx):
        sp_set, sp_labels, q_set, q_labels, classes = batch
        
        pred_logits = self.model(sp_set, sp_labels, q_set)
        q_labels = q_labels.to(pred_logits.device)

        res = criterion(pred_logits, q_labels)

        self.log('train_loss', res['loss'], on_step=True, prog_bar=True)
        self.log('train_acc', res['acc'], on_step=True, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        sp_set, sp_labels, q_set, q_labels, classes = batch

        pred_logits = self.model(sp_set, sp_labels, q_set)
        q_labels = q_labels.to(pred_logits.device)
        
        res = criterion(pred_logits, q_labels)
        
        self.log('val_loss', res['loss'], on_step=True, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('val_acc', res['acc'], on_step=True, on_epoch=True, batch_size=1, prog_bar=True)
    
    def on_test_start(self):
        self.accuracies = []
    
    def on_test_end(self):
        avg_acc, confidence_interval = mean_confidence_interval(self.accuracies)
        print(f'Accuracy: {avg_acc} +- {confidence_interval}')

    def test_step(self, batch, batch_idx):
        sp_set, sp_labels, q_set, q_labels, classes = batch        
        
        pred_logits = self.model(sp_set, sp_labels, q_set)
        q_labels = q_labels.to(pred_logits.device)

        acc = accuracy(pred_logits, q_labels, calc_mean=False)
        self.accuracies += acc.tolist()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.trainer.model.parameters(), lr=self.cfg['lr'])
        return optimizer

def get_dataloader(dataloader_cfg):
    dataloader = {}
    for name, cfg in dataloader_cfg.items():
        dataset = UCF_TRX(cfg['dataset'])
        dataloader[name] = get_episodic_dataloader(dataset, **cfg['sampler'])
    return dataloader

if __name__ == '__main__':
    print(torch.cuda.is_available())

    dataloader = get_dataloader(cfg.dataloader)
    
    model = CNN_TRX(cfg.model)
    
    pl_model = TRXModel(model, cfg.plmodule)
    #
    with open('secrets.yaml', 'r') as f:
        secrets = yaml.safe_load(f)
    os.environ["WANDB_API_KEY"] = secrets['wandb_api_key']
    # logger = pl.loggers.WandbLogger(**cfg.logger)
    logger = None
    
    workdir = get_workdir("workdirs/trx")
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='val_acc',
        mode='max',
        save_last=True,
        dirpath=workdir,
    )
    print('SLURM_NTASKS =', os.environ['SLURM_NTASKS'])
    print('SLURM_TASKS_PER_NODE =', os.environ['SLURM_TASKS_PER_NODE'])
    print('SLURM_NNODES =', os.environ['SLURM_NNODES'])

    
    trainer_cfg = edict(
        devices=2,
        accelerator='gpu',
        strategy='fsdp',
        
        max_epochs=1,
        
        check_val_every_n_epoch=None,
        val_check_interval=1000,

        accumulate_grad_batches=8,
        
        logger = logger,
        callbacks = [ckpt_cb]
    )
    trainer = pl.Trainer(**trainer_cfg)
    trainer.fit(pl_model, dataloader['train'], dataloader['val'])
    trainer.test(pl_model, dataloader['test'])





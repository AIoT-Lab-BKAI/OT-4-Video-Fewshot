import os
from models.MOLO import CNN_BiMHM_MoLo
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import pytorch_lightning as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam

from utils.utils import criterion, get_wandb_logger, get_workdir, mean_confidence_interval, accuracy

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from datasets.base.builder import build_fewshot_dataset

import yaml
import torch.nn.functional as F
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

class RunningAverage(object):
    def __init__(self, size=100):
        self.list = []
        self.size = size
    def append(self, val):
        self.list.append(val)
        if len(self.list) > self.size:
            self.list.pop(0)
    def get_avg(self):
        return sum(self.list) / len(self.list)

class BasicModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.automatic_optimization = False
        self.train_loss_ra = RunningAverage(cfg.get('log_freq', 100))
        self.train_acc_ra = RunningAverage(cfg.get('log_freq', 100))
    
    def criterion(self, input):
        query_labels = input['query_labels']
        query_labels = query_labels.view(-1).long()
        
        loss = F.cross_entropy(input["logits"], query_labels)
        loss += self.cfg['recons_coef'] * input['loss_recons']
        
        coef = self.cfg['contrastive_coef']
        loss += coef * F.cross_entropy(input["logits_s2q"], query_labels) 
        loss += coef * F.cross_entropy(input["logits_q2s"], query_labels)
        loss += coef * F.cross_entropy(input["logits_s2q_motion"], query_labels)
        loss += coef * F.cross_entropy(input["logits_q2s_motion"], query_labels)

        acc = accuracy(input['logits'], query_labels)
        return {"loss": loss, "acc":acc}
    
    def training_step(self, batch, batch_idx):  
        batch['train'] = True      
        predict = self.model(batch)
        res = self.criterion(predict)
        
        self.train_loss_ra.append(res['loss'].item())
        self.train_acc_ra.append(res['acc'].item())

        self.manual_backward(res['loss']/self.cfg['accum_grad'])

        if batch_idx % self.cfg['accum_grad'] == 0:
            opt = self.optimizers()
            opt.zero_grad()
            opt.step()
        
        if batch_idx % self.cfg['log_freq'] == 0:
            self.log('train_loss', self.train_loss_ra.get_avg(), on_step=True, prog_bar=True)
            self.log('train_acc', self.train_acc_ra.get_avg(), on_step=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        predict = self.model(batch)
        res = self.criterion(predict)
                
        self.log('val_loss', res['loss'], on_epoch=True, batch_size=1, prog_bar=True)
        self.log('val_acc', res['acc'], on_epoch=True, batch_size=1, prog_bar=True)
    
    def on_test_start(self):
        self.accuracies = []

    def on_train_start(self):
        # torch.cuda.empty_cache()
        pass
    
    def on_test_end(self):
        avg_acc, confidence_interval = mean_confidence_interval(self.accuracies)
        print(f'Accuracy: {avg_acc} +- {confidence_interval}')

    def test_step(self, batch, batch_idx):        
        predict = self.model(batch)

        acc = accuracy(predict['logits'], predict['query_labels'], calc_mean=False)
        self.accuracies += acc.tolist()
    
    def configure_optimizers(self):
        bn_params = []                  # Batchnorm parameters.
        head_parameters = []            # Head parameters
        non_bn_parameters = []          # Non-batchnorm parameters.
        no_weight_decay_parameters = [] # No weight decay parameters
        for name, p in self.model.named_parameters():
            if "embd" in name or "cls_token" in name:
                no_weight_decay_parameters.append(p)
            elif "bn" in name or "norm" in name:
                bn_params.append(p)
            elif "head" in name:
                head_parameters.append(p)
            else:
                non_bn_parameters.append(p)
        optim_params = [
            {"params": non_bn_parameters, "weight_decay": self.cfg['optimizer']['weight_decay']},
            {"params": head_parameters, "weight_decay": self.cfg['optimizer']['weight_decay']},
            {"params": no_weight_decay_parameters, "weight_decay": 0.0},
            {"params": bn_params, "weight_decay": self.cfg['bn']['weight_decay']},
        ]
        optimizer = DeepSpeedCPUAdam(optim_params, lr=0.0001)

        return optimizer

def get_dataloader(cfg):
    dataloader = {}
    for name, ds_cfg in cfg.dataset.items():
        dataset = build_fewshot_dataset("Ucf101", ds_cfg, cfg.sampler[name])
        dataloader[name] = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--cfg', type=str, default='configs/MoLo.py')
    parser.add_argument('--local_rank', type=int, default=-1)
    return parser.parse_args()

def get_callbacks(cfg):
    callbacks = []
    if hasattr(cfg, 'checkpoint'):
        wkdir = get_workdir("workdirs/MoLo")
        ckpt_cb = pl.callbacks.ModelCheckpoint(
            **cfg.checkpoint, dirpath=wkdir)
        callbacks.append(ckpt_cb)
    if hasattr(cfg, 'lr_monitor'):
        lr_monitor = pl.callbacks.LearningRateMonitor(
            **cfg.lr_monitor)
        callbacks.append(lr_monitor)
    return callbacks

if __name__ == '__main__':
    args = parse_args()

    import importlib
    cfg = importlib.import_module(args.cfg.replace('/', '.').replace('.py', ''))
    
    dataloader = get_dataloader(cfg)

    model1 = CNN_BiMHM_MoLo(cfg.model)
    model = BasicModule(model1, cfg.pl_module)
    model = load_state_dict_from_zero_checkpoint(model, 'workdirs/MoLo/08_21_17_29/epoch=0-step=1250.ckpt')



    if args.log:
        logger = get_wandb_logger(cfg.logger)
    else:
        logger = None
    
    cbs = get_callbacks(cfg)

    trainer = pl.Trainer(**cfg.trainer, logger=logger, callbacks=cbs)
    # trainer.fit(model, dataloader['train'], dataloader['val'])
    trainer.test(model, dataloader['test'])

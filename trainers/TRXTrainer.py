import os
from datasets.UCF_TRX import UCF_TRX
from datasets.UCF_model1 import FS_DS
from models.TRX import CNN_TRX
import pytorch_lightning as pl

from utils.utils import criterion, get_episodic_dataloader, get_wandb_logger, get_workdir, mean_confidence_interval, accuracy
import yaml
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from argparse import ArgumentParser
from torch.utils.data import DataLoader

from deepspeed.ops.adam import DeepSpeedCPUAdam
# from models import MODEL_REGISTRY

def preprocess(batch):
    batch = list(batch)
    for i in range(len(batch)):
        batch[i] = batch[i].squeeze(0)
    return batch

class BasicModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def training_step(self, batch, batch_idx):
        batch = preprocess(batch)
        sp_set, sp_labels, q_set, q_labels = batch
        
        pred_logits = self.model(sp_set, sp_labels, q_set)
        
        q_labels = q_labels.to(pred_logits.device)
        res = criterion(pred_logits, q_labels)

        self.log('train_loss', res['loss'], on_step=True, prog_bar=True)
        self.log('train_acc', res['acc'], on_step=True, prog_bar=True)

        return res['loss']
    
    def validation_step(self, batch, batch_idx):
        batch = preprocess(batch)
        sp_set, sp_labels, q_set, q_labels = batch

        pred_logits = self.model(sp_set, sp_labels, q_set)
        
        q_labels = q_labels.to(pred_logits.device)
        res = criterion(pred_logits, q_labels)
        
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
        batch = preprocess(batch)
        sp_set, sp_labels, q_set, q_labels = batch        
        
        pred_logits = self.model(sp_set, sp_labels, q_set)
        q_labels = q_labels.to(pred_logits.device)

        acc = accuracy(pred_logits, q_labels, calc_mean=False)
        self.accuracies += acc.tolist()
    
    def configure_optimizers(self):
        optimizer = DeepSpeedCPUAdam(self.model.parameters())

        return optimizer

def get_dataloader():
    dataloader = {}
    for name, ds_cfg in cfg.dataset.items():
        dataset = UCF_TRX(**ds_cfg)
        dataset = FS_DS(dataset, **cfg.sampler[name])
        dataloader[name] = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    return dataloader

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--cfg', type=str, default='configs/TRX.py')
    parser.add_argument('--local_rank', type=int, default=-1)
    return parser.parse_args()

def get_callbacks():
    wkdir = get_workdir("workdirs/trx")
    cbs = []
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        **cfg.checkpoint, dirpath=wkdir)
    cbs.append(ckpt_cb)
    if hasattr(cfg, 'lr_monitor'):
        lr_monitor = pl.callbacks.LearningRateMonitor(
            **cfg.lr_monitor)
        cbs.append(lr_monitor)
    return cbs
cfg = None
if __name__ == '__main__':
    args = parse_args()

    import importlib
    cfg = importlib.import_module(args.cfg.replace('/', '.').replace('.py', ''))

    dataloader = get_dataloader()

    # model = CNN_TRX(cfg.model)
    model = M1(**cfg.model)
    pl_model = TRXModel(model, cfg.plmodule)

    if args.log:
        logger = get_wandb_logger(cfg.logger)
    else:
        logger = None
    
    cbs = get_callbacks()

    trainer = pl.Trainer(**cfg.trainer, logger=logger, callbacks=cbs)
    trainer.fit(pl_model, dataloader['train'], dataloader['val'])
    trainer.test(pl_model, dataloader['test'])







from datasets.UCF_model1 import UCF101_M1
from models.M1 import Model1
from utils.utils import accuracy, criterion, get_episodic_dataloader, mean_confidence_interval
import pytorch_lightning as pl
import torch
import configs.m1 as CFG
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class BasicTrainer(pl.LightningModule):
    def __init__(self, model, opt_cfg):
        super().__init__()
        self.model = model
        self.opt_cfg = opt_cfg
    
    def training_step(self, batch, batch_idx):
        sp_set, sp_labels, q_set, q_labels, _ = batch
        
        pred_logits = self.model(sp_set, sp_labels, q_set)
        
        res = criterion(pred_logits, q_labels)

        self.log('train_loss', res['loss'], on_step=True, prog_bar=True)
        self.log('train_acc', res['acc'], on_step=True, prog_bar=True)

        return res['loss']
   
    def validation_step(self, batch, batch_idx):
        sp_set, sp_labels, q_set, q_labels, classes = batch
        pred_logits = self.model(sp_set, sp_labels, q_set)
        
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

        acc = accuracy(pred_logits, q_labels, calc_mean=False)
        self.accuracies += acc.tolist()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.trainer.model.parameters(), **self.opt_cfg)
        
        return optimizer

if __name__ == '__main__':
    # get dataset
    dataloader = {}
    for name, cfg in CFG.dataloader.items():
        dataset = UCF101_M1(**cfg['dataset'])
        dataloader[name] = get_episodic_dataloader(dataset, **cfg['sampler'])

    # get model
    model = Model1(**CFG.model)
    # get trainer
    pl_module = BasicTrainer(model, CFG.optimizer)
    # trainer
    trainer = pl.Trainer(**CFG.trainer)
    trainer.fit(pl_module, dataloader['train'], dataloader['val'])
    trainer.test(pl_module, dataloader['test'])

import sys
sys.path.append('/vinserver_user/bach.vv200061/optimal-transport-c3d')
import lightning.pytorch as pl
import torch
from easydict import EasyDict as edict
from utils.utils import accuracy, get_checkpoint_callback, get_episodic_dataloader, get_wandb_logger, mean_confidence_interval
from datasets.UCF101 import UCF101_C3D
from models.C3D import C3D
from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
import torch.nn as nn
import json




class C3DLightning(pl.LightningModule):
    def __init__(self, model_cfg, cfg):
        super().__init__()
        self.model = C3D(model_cfg)
        
        if hasattr(model_cfg, 'ckpt'):
            ckpt = torch.load(model_cfg.ckpt)
            self.model.load_state_dict(ckpt['state_dict'])
        
        self.cfg = cfg
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_fn = accuracy
        
        self.automatic_optimization = False
 
    def training_step(self, batch, batch_idx):
        sp_set, sp_labels, q_set, q_labels, _ = batch
        q_labels = torch.repeat_interleave(torch.arange(self.cfg.n_way), self.cfg.n_query).to(sp_set.device)
        
        pred_logits = self.model(sp_set, q_set)
        
        loss = self.loss_fn(pred_logits, q_labels) / self.cfg.gradient_accumulation_steps
        
        self.manual_backward(loss)
        
        if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
            optimizer = self.optimizers()
            optimizer.step()
            optimizer.zero_grad()

        accuracy = self.accuracy_fn(pred_logits, q_labels)
        
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        sp_set, sp_labels, q_set, q_labels, _ = batch
        q_labels = torch.repeat_interleave(torch.arange(self.cfg.n_way), self.cfg.n_query).to(sp_set.device)

        pred_logits = self.model(sp_set, q_set)

        loss = self.loss_fn(pred_logits, q_labels)
        accuracy = self.accuracy_fn(pred_logits, q_labels)
        
        self.log('val_loss', loss, prog_bar=True, batch_size=self.cfg.batch_size)
        self.log('val_acc', accuracy, prog_bar=True, batch_size=self.cfg.batch_size)
    
    def on_test_start(self):
        self.accuracies = []
    
    def on_test_end(self):
        avg_acc, confidence_interval = mean_confidence_interval(self.accuracies)
        print(f'Accuracy: {avg_acc} +- {confidence_interval}')

    def test_step(self, batch, batch_idx):
        sp_set, sp_labels, q_set, q_labels, classes = batch
        q_labels = torch.repeat_interleave(torch.arange(self.cfg.n_way), self.cfg.n_query).to(sp_set.device)
        
        pred_logits = self.model(sp_set, q_set)

        accuracy = self.accuracy_fn(pred_logits, q_labels, calc_mean=False)
        self.accuracies += accuracy.tolist()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        lr_scheduler = MultiStepLR(optimizer, milestones=self.cfg.lr_decay_step, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


if __name__ == '__main__':
    class_path = dict(train='config/train.json', val='config/val.json', test='config/test.json')
    dataset_general_cfg = edict(
        n_seg = 4,
        seglen = 16,
        img_size = 112,
        root_dir = 'data/ucf101',
        class_id = json.load(open('splits/classes.json')),
        n_way = 5,
        n_shot = 1,
        n_query = 3
    )
    dataset_cfg = edict(
        train = edict(
            **dataset_general_cfg,
            classes = json.load(open('splits/train.json')),
        ),
        val = edict(
            **dataset_general_cfg,
            classes = json.load(open('splits/val.json')),
        ),
        test = edict(
            **dataset_general_cfg,
            classes = json.load(open('splits/test.json')),
        )
    )
    dataloader_cfg = edict(
        train = 20000,
        val = 500,
        test = 10000
    )
    model_cfg = edict(
        use_positional_cost = False,
        entropic_reg = 0.1,
        **dataset_general_cfg
    )
    training_cfg = edict(
        batch_size=1,
        learning_rate = 1e-3,
        lr_decay_step = [100000],
        gradient_accumulation_steps = 4,
        **dataset_general_cfg        
    )
    trainer_cfg = edict(
        devices=1,
        accelerator='gpu',
        max_epochs=1,
        check_val_every_n_epoch=None,
        val_check_interval=500,
    )
    logger_cfg = {
        'entity': 'aiotlab',
        'project': 'few-shot-action-recognition',
        'group': 'trx'
    }
    ckpt_cfg = edict(
        save_top_k=1,
        monitor='val_acc',
        mode='min',
        save_last=True,
        dirpath='workdirs/c3d'
    )

    dataset = {}
    dataloader = {}
    for name in ['train', 'val', 'test']:
        dataset[name] = UCF101_C3D(dataset_cfg[name])
        n_way, n_shot, n_query = dataset_general_cfg.n_way, dataset_general_cfg.n_shot, dataset_general_cfg.n_query
        sampler = TaskSampler(dataset[name], n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=dataloader_cfg[name])
        dataloader[name] = DataLoader(dataset[name], batch_sampler=sampler, num_workers=2, pin_memory=True, collate_fn=sampler.episodic_collate_fn)
    
    model = C3DLightning(model_cfg, training_cfg)

    logger = get_wandb_logger(logger_cfg)
    ckpt_callback = pl.callbacks.ModelCheckpoint(**ckpt_cfg)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(**trainer_cfg, logger=logger, callbacks=[ckpt_callback, lr_monitor])
    trainer.fit(model, dataloader['train'], dataloader['val'])
    trainer.test(model, dataloader['test'])






    

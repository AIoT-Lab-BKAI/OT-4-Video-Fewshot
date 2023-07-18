import json
import os
from datasets.TRX_Dataset import TRX_Dataset
from models.TRX import CNN_TRX
from torch.optim.lr_scheduler import MultiStepLR
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import lightning.pytorch as pl
from utils.utils import get_episodic_dataloader, mean_confidence_interval
import yaml

def accuracy(logits, labels, calc_mean=True):
    correct = torch.argmax(logits, 1) == labels
    if calc_mean:
        return torch.mean(correct.float())
    else:
        return correct.float()

def collect_support_labels():
    pass 

class TRXModel(pl.LightningModule):
    def __init__(self, model_cfg, cfg):
        super().__init__()
        self.model = CNN_TRX(**model_cfg)
        self.cfg = edict(cfg)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_fn = accuracy
        self.automatic_optimization = False
 
    def training_step(self, batch, batch_idx):
        sp_set, sp_labels, q_set, q_labels, classes = batch
        sp_labels = torch.arange(self.cfg.n_way).to(sp_set.device)
        q_labels = torch.repeat_interleave(torch.arange(self.cfg.n_way), self.cfg.n_query).to(sp_set.device)
        
        pred_logits = self.model(sp_set, sp_labels, q_set)
        pred_logits = pred_logits.squeeze()
        loss = self.loss_fn(pred_logits, q_labels) / self.cfg.gradient_accumulation_steps
        self.manual_backward(loss)
        if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
            optimizer = self.optimizers()
            lr_scheduler = self.lr_schedulers()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        accuracy = self.accuracy_fn(pred_logits, q_labels)
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        sp_set, sp_labels, q_set, q_labels, classes = batch
        sp_labels = torch.arange(self.cfg.n_way).to(sp_set.device)
        q_labels = torch.repeat_interleave(torch.arange(self.cfg.n_way), self.cfg.n_query).to(sp_set.device)

        pred_logits = self.model(sp_set, sp_labels, q_set)
        pred_logits = pred_logits.squeeze()

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
        sp_labels = torch.arange(self.cfg.n_way).to(sp_set.device)
        q_labels = torch.repeat_interleave(torch.arange(self.cfg.n_way), self.cfg.n_query).to(sp_set.device)
        
        pred_logits = self.model(sp_set, sp_labels, q_set)
        pred_logits = pred_logits.squeeze()

        accuracy = self.accuracy_fn(pred_logits, q_labels, calc_mean=False)
        self.accuracies += accuracy.tolist()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        lr_scheduler = MultiStepLR(optimizer, milestones=self.cfg.lr_decay_step, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

def get_fake_data():
    sp_set = torch.randn((5, 16, 3, 112, 112))
    q_set = torch.randn((25, 16, 3, 112, 112))
    sp_labels = None
    q_labels = None
    classes = None
    batch = (sp_set, sp_labels, q_set, q_labels, classes)

def get_dataset_dataloader(dataset_cfg):
    class_id = json.load(open(dataset_cfg.train.class_id_path, 'r'))
    dataset = edict()
    dataloader = edict()
    for name in dataset_cfg:
        cfg = dataset_cfg[name]
        classes = json.load(open(cfg.class_path, 'r'))
        dataset[name] = TRX_Dataset(cfg.data_dir, class_id, classes, cfg.seq_len)
        dataloader[name] = get_episodic_dataloader(dataset[name], cfg.n_way, cfg.n_shot, cfg.n_query, cfg.n_tasks)
    return dataset, dataloader

# init the autoencoder
if __name__ == '__main__':

    class_path = dict(train='config/train.json', val='config/val.json', test='config/test.json')
    dataset_general_cfg = edict(
        n_way = 5,
        n_shot = 1,
        n_query = 5,
        seq_len = 16,
        img_size = 112,
        data_dir = 'data/ucf101',
        class_id_path = 'config/classes.json'
    )
    dataset_cfg = edict(
        train = edict(
            **dataset_general_cfg,
            class_path = 'config/train.json',
            n_tasks = 20000,
        ),
        val = edict(
            **dataset_general_cfg,
            class_path = 'config/val.json',
            n_tasks = 1000,
        ),
        test = edict(
            **dataset_general_cfg,
            class_path = 'config/test.json',
            n_tasks = 2000,
        )
    )
    model_cfg = dict(args=edict(dict(
        trans_linear_in_dim = 512,
        trans_linear_out_dim = 128,
        trans_dropout = 0.1,
        way = dataset_general_cfg.n_way,
        shot = dataset_general_cfg.n_shot,
        query_per_class = dataset_general_cfg.n_query,
        seq_len = dataset_general_cfg.seq_len,
        img_size = dataset_general_cfg.img_size,
        method = "resnet18",
        num_gpus = 1,
        temp_set = [2,3]
    )))
    training_cfg = edict(
        n_way = dataset_general_cfg.n_way,
        n_shot = dataset_general_cfg.n_shot,
        n_query = dataset_general_cfg.n_query,
        learning_rate = 1e-3,
        lr_decay_step = [100000],
        gradient_accumulation_steps = 4,
        batch_size = 1,
    )
    trainer_cfg = edict(
        devices=1,
        accelerator='gpu',
        max_epochs=1,
        check_val_every_n_epoch=None,
        val_check_interval=2000
    )
    logger_cfg = {
        'entity': 'aiotlab',
        'project': 'few-shot-action-recognition',
        'group': 'trx'
    }

    dataset, dataloader = get_dataset_dataloader(dataset_cfg)

    model = TRXModel(model_cfg, training_cfg)
    
    with open('secrets.yaml', 'r') as f:
        secrets = yaml.safe_load(f)
    os.environ["WANDB_API_KEY"] = secrets['wandb_api_key']
    logger = pl.loggers.WandbLogger(**logger_cfg)

    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True,
        dirpath='workdirs/trx',
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(**trainer_cfg, logger=logger, callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(model, dataloader['train'], dataloader['val'])
    trainer.test(model, dataloader['test'])





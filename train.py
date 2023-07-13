import os
import random
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import UCF101Dataset
from model import C3DModel
import json
from tqdm import tqdm
import numpy as np
import scipy
import yaml
from easyfsl.samplers import TaskSampler
import argparse 

def parser_args():
    pass

def to_device(data, device):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device)
    return data

def train_loss(predict_distance, support_labels, query_labels):
    n_query = query_labels.shape[0]
    n_way = support_labels.shape[0]    
    loss = 0
    log_softmax = torch.nn.functional.log_softmax(-predict_distance, dim=1)
    for query in range(n_query):
        for support in range(n_way):
            if support_labels[support] == query_labels[query]:
                loss += -log_softmax[query, support]

    return loss

def accuracy(predict, groundtruth):
    predict = predict.reshape(-1)
    groundtruth = groundtruth.reshape(-1)
    accs = torch.where(predict == groundtruth, 1.0, 0.0)
    accs = accs.cpu().tolist()
    return accs

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

N_WAY = 5
K_SHOT = 5
N_QUERY = 10  # Number of images per class in the query set

class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        # prepare dataset, dataloader        
        dataset = {}
        dataloader = {}
        for ds_type in cfg['dataset']:
            ds_cfg = cfg['dataset'][ds_type]
            classes = json.load(open(ds_cfg['class_path']))
            dataset[ds_type] = UCF101Dataset(ds_cfg['root_dir'], classes)
            
            sampler = TaskSampler(dataset[ds_type], n_way=cfg['n_way'], n_shot=cfg['k_shot'], n_query=cfg['n_query'], n_tasks=ds_cfg['num_episodes'])
            dataloader[ds_type] = DataLoader(dataset[ds_type], batch_sampler=sampler, num_workers=2, pin_memory=True, collate_fn=sampler.episodic_collate_fn)        
        
        self.dataset = dataset
        self.dataloader = dataloader
        
        # prepare model
        self.model = C3DModel(cfg).to(cfg['device'])
        if hasattr(cfg, 'pretrained') and cfg['pretrained']:
            ckpt = torch.load(cfg['pretrained'])
            self.model.load_state_dict(ckpt['state_dict'])
        # prepare optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # prepare logging
        wandb_cfg = cfg['wandb']
        self.log = cfg['log']
        if self.log:
            with open(wandb_cfg['secret_file'], 'r') as f:
                secrets = yaml.safe_load(f)
            os.environ["WANDB_API_KEY"] = secrets['wandb_api_key']
            wandb.init(project=wandb_cfg['project'], group=wandb_cfg['group'])
    
    def train(self):
        loader = self.dataloader['train']
        for id, data in tqdm(enumerate(loader)):
            data[-1] = torch.tensor(data[-1])
            for i in range(len(data)):
                data[i] = data[i].to(self.cfg['device'])
            sp_set, sp_labels, q_set, q_labels, classes = data
            predict_distance = self.model(sp_set, q_set)     
            loss = train_loss(predict_distance, torch.unique_consecutive(sp_labels), q_labels)
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            print(loss)
            if self.log:
                wandb.log({'train_loss': loss})
            
            # evaluation 
            if id % self.cfg['eval_interval'] == 0 and id > 0:
                val_loss = self.eval()
                self.save_best(val_loss)
                self.model.train()
                if self.log:
                    wandb.log({'val_loss': val_loss})
            
            if id % self.cfg['ckpt_interval'] == 0 and id > 0:
                t = id / self.cfg['ckpt_interval']
                ckpt_path = os.path.join(self.cfg['workdir'], f'ckpt_{t}.pth')
                torch.save({'state_dict': self.model.state_dict()}, ckpt_path)
        
        self.save_last()
    
    def eval(self):
        model = self.model
        model.eval()
        
        loader = self.dataloader['val']
        total_len = len(loader)
        mean_loss = 0
        with torch.no_grad():
            for id, data in tqdm(enumerate(loader)):
                data[-1] = torch.tensor(data[-1])
                for i in range(len(data)):
                    data[i] = data[i].to(self.cfg['device'])
                
                sp_set, sp_labels, q_set, q_labels, classes = data
                predict_distance = self.model(sp_set, q_set)     
                mean_loss = train_loss(predict_distance, torch.unique_consecutive(sp_labels), q_labels)/total_len
        
        return mean_loss
    
    def get_predict_labels(self, predict_distance, support_labels):
        n_query = predict_distance.shape[0]
        predict_labels = torch.zeros(n_query).to(predict_distance.device)
        for query in range(n_query):
            predict_labels[query] = support_labels[torch.argmin(predict_distance[query])]
    
        return predict_labels
    
    def test(self):
        model = self.model
        if hasattr(self, 'best_model') and self.best_model:
            ckpt = torch.load(self.best_model)    
            model.load_state_dict(ckpt['state_dict'])
        
        model.eval()
        
        loader = self.dataloader['test']
        accuracy_list = []
        with torch.no_grad():
            for id, data in tqdm(enumerate(loader)):
                data[-1] = torch.tensor(data[-1])
                for i in range(len(data)):
                    data[i] = data[i].to(self.cfg['device'])
                
                sp_set, sp_labels, q_set, q_labels, classes = data
                predict_distance = self.model(sp_set, q_set)
                predict_labels = self.get_predict_labels(predict_distance, torch.unique_consecutive(sp_labels))     
                
                acc = accuracy(predict_labels, q_labels)
                accuracy_list += acc
        
        mean_acc, confidence_interval = mean_confidence_interval(accuracy_list)
        print(mean_acc, confidence_interval)
        if self.log:
            wandb.log({'mean_accuracy': mean_acc, 'confidence_interval': confidence_interval})
        
    def save_best(self, val_loss):
        if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            wdir = self.cfg['workdir']
            self.best_model = f'{wdir}/best_model.pth'
            torch.save({'state_dict': self.model.state_dict()}, self.best_model)
    
    def save_last(self):
        wdir = self.cfg['workdir']
        self.last_model = f'{wdir}/last_model.pth'        
        torch.save({'state_dict': self.model.state_dict()}, self.last_model)
   

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, help="path to config file", default='opt.yaml')      
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    
    trainer = Trainer(cfg)
    trainer.train()
    trainer.test()

if __name__ == '__main__':
    main()
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
def parser_args():
    pass

def to_device(data, device):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device)
    return data

def train_loss(predict_distance, support_labels, query_labels):
    nbatch = predict_distance.shape[0]
    nquery = predict_distance.shape[1]
    nway = predict_distance.shape[2]
    
    loss = 0
    log_softmax = torch.nn.functional.log_softmax(-predict_distance, dim=2)
    for batch in range(nbatch):
        for query in range(nquery):
            for support in range(nway):
                if support_labels[batch, support] == query_labels[batch, query]:
                    loss += -log_softmax[batch, query, support]

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


class Trainer():
    def __init__(self):
        self.device = 'cuda'
        self.workdir = 'workdirs'

        # prepare dataset, dataloader
        classes = {}
        class_path = {'train': 'ucf101/train.json', 'test': 'ucf101/test.json', 'val': 'ucf101/val.json'}
        for key, path in class_path.items():
            classes[key] = json.load(open(path, 'r'))
        
        dataset_root = 'data/ucf101'
        n_iter = {'train': 10000, 'val': 1000, 'test':2000}
        self.eval_iter = 500
        nway, kshot = 5, 1
        dataset = {}
        dataloader = {}
        for type in ['train', 'val', 'test']:
            dataset[type] = UCF101Dataset(dataset_root, classes[type], n_iter[type], n_way=nway, k_shot=kshot)
            dataloader[type] = DataLoader(dataset[type], batch_size=1, shuffle=(True if type=='Train' else False), num_workers=4)
        self.dataset = dataset
        self.dataloader = dataloader
        # prepare model
        self.nseg = 4
        self.model = C3DModel(self.nseg).to(self.device)
        ckpt = torch.load('pretrained/c3d_sports1m-pretrained.pt')
        self.model.load_state_dict(ckpt['state_dict'])
        
        # prepare optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # prepare logging
        project_name = 'optimal_transport'
        run_group = 'c3d'
        with open('secret_key.yaml', 'r') as f:
            secrets = yaml.safe_load(f)
        os.environ["WANDB_API_KEY"] = secrets['wandb_api_key']
        self.log = True
        if self.log:
            wandb.init(project=project_name, group=run_group)


    def train(self):
        loader = self.dataloader['train']
        for id, data in tqdm(enumerate(loader)):
            data = to_device(data, self.device)

            predict_distance = self.model(data['support_set'], data['query_set'])     

            loss = train_loss(predict_distance, data['support_labels'], data['query_labels'])
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(loss)
            if self.log:
                wandb.log({'train_loss': loss})
            

            # evaluation 
            if id % self.eval_iter == 0 and id > 0:
                val_loss = self.eval()
                self.save_best(val_loss)
                self.model.train()
                if self.log:
                    wandb.log({'val_loss': val_loss})
        
        self.save_last()
    
    def eval(self):
        model = self.model
        model.eval()
        loader = self.dataloader['val']
        total_len = len(loader)
        total_loss = 0
        with torch.no_grad():
            for id, data in tqdm(enumerate(loader)):
                data = to_device(data, self.device)
                predict_distance = model(data['support_set'], data['query_set'])
                total_loss += train_loss(predict_distance, data['support_labels'], data['query_labels'])/total_len
        return total_loss
    
    def get_predict_labels(self, predict_distance, support_labels):
        nbatch = predict_distance.shape[0]
        nquery = predict_distance.shape[1]
        predict_labels = torch.zeros(nbatch, nquery).to(self.device)
        for batch in range(nbatch):
            for query in range(nquery):
                predict_labels[batch, query] = support_labels[batch, torch.argmin(predict_distance[batch, query])]
    
        return predict_labels
    
    def test(self):
        model = self.model
        loader = self.dataloader['test']
        if hasattr(self, 'best_model') and self.best_model:
            ckpt = torch.load(self.best_model)
        
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        accuracy_list = []
        with torch.no_grad():
            for id, data in tqdm(enumerate(loader)):
                data = to_device(data, self.device)
                predict_distance = model(data['support_set'], data['query_set'])
                predict_labels = self.get_predict_labels(predict_distance, data['support_labels'])
                acc = accuracy(predict_labels, data['query_labels'])
                accuracy_list += acc
        
        mean_acc, confidence_interval = mean_confidence_interval(accuracy_list)
        if self.log:
            wandb.log({'mean_accuracy': mean_acc, 'confidence_interval': confidence_interval})
        
    def save_best(self, val_loss):
        if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model = f'{self.workdir}/best_model.pth'
            torch.save({'state_dict': self.model.state_dict()}, self.best_model)
    
    def save_last(self):
        self.last_model = f'{self.workdir}/last_model.pth'
        torch.save({'state_dict': self.model.state_dict()}, self.last_model)
                
def main():
    trainer = Trainer()
    trainer.train()
    trainer.test()

if __name__ == '__main__':
    main()
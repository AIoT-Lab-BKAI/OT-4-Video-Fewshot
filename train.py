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
    nbatch = support_labels.shape[0]
    nway = support_labels.shape[1]
    nquery = query_labels.shape[1]
    
    loss = 0
    sum_distance = torch.sum(torch.exp(-predict_distance), dim=2)
    for batch in range(nbatch):
        for query in range(nquery):
            for support in range(nway):
                if support_labels[batch, support] == query_labels[batch, query]:
                    loss += -torch.log(torch.exp(-predict_distance[batch, query, support])/sum_distance[batch, query] + 1e-4)
    
    return loss

def accuracy(predict, groundtruth):
    predict = predict.reshpae(-1)
    groundtruth = groundtruth.reshape(-1)
    accs = torch.where(predict == groundtruth, 1.0, 0.0)
    accs = torch.cpu().tolist()
    return accs

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h


class Trainer():
    def __init__(self):
        self.device = 'cuda:0'
        workdir = 'workdirs'
        # prepare dataset, dataloader
        classes = {}
        class_path = {'train': 'ucf101/train.json', 'test': 'ucf101/test.json', 'val': 'ucf101/val.json'}
        for key, path in class_path.items():
            classes[key] = json.load(open(path, 'r'))
        
        dataset_root = 'data/ucf101'
        n_iter = {'train': 1, 'val': 1000, 'test':2000}
        nway, kshot = 5, 5
        dataset = {}
        dataloader = {}
        for type in ['train', 'val', 'test']:
            dataset[type] = UCF101Dataset(dataset_root, classes[type], n_iter[type], n_way=nway, k_shot=kshot)
            dataloader[type] = DataLoader(dataset[type], batch_size=2, shuffle=(True if type=='Train' else False), num_workers=4)
        
        # prepare model
        self.nseg = 4
        self.model = C3DModel(self.nseg).to(self.device)
        ckpt = torch.load('c3d_sports1m-pretrained.pt')
        self.model.load_state_dict(ckpt['state_dict'])
        
        # prepare optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # prepare logging
        self.eval_iter = 500
        project_name = 'optimal_transport'
        run_group = 'c3d'
        with open('secrets.yaml', 'r') as f:
            secrets = yaml.load(f)
        os.environ["WANDB_API_KEY"] = secrets['wandb_api_key']
        # wandb.init(project=project_name, group=run_group)


    def train(self):
        loader = self.dataloader['train']
        import pdb; pdb.set_trace()
        for id, data in tqdm(enumerate(loader)):
            data = to_device(data, self.device)
            predict_distance = self.model(data['support_set'], data['support_labels'], data['query_set'])     
            
            loss = train_loss(predict_distance, data['support_labels'], data['query_labels'])
            loss.backward()
            
            self.optimizer.step()

            # wandb.log({'train_loss': loss})
            
            # evaluation 
            if id % self.eval_iter == 0 and id > 0:
                val_loss = self.eval()
                # wandb.log({'val_loss': val_loss, 'eval_iter':id})
                self.save_best(val_loss)
                self.model.train()
        
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
                predict_distance = model(data['support_set'], data['support_labels'], data['query_set'])
                total_loss += train_loss(predict_distance, data['support_labels'], data['query_labels'])/total_len
        return total_loss
    
    def get_predict_labels(self, predict_distance, support_labels):
        nbatch = predict_distance.shape[0]
        nquery = predict_distance.shape[1]
        predict_labels = torch.zeros(nbatch, nquery)
        for batch in range(nbatch):
            for query in range(nquery):
                predict_labels[batch, query] = support_labels[torch.argmin(predict_distance[batch, query])]
    
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
                predict_distance = model(data['support_set'], data['support_labels'], data['query_set'])
                predict_labels = self.get_predict_labels(predict_distance, data['support_labels'])
                acc = accuracy(predict_labels, data['query_labels'])
                accuracy_list += acc
        mean_accuracy = sum(accuracy_list)/len(accuracy_list)
        confidence_interval = mean_confidence_interval(accuracy_list)
        wandb.log({'mean_accuracy': mean_accuracy, 'confidence_interval': confidence_interval})
        
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
    
    
    # # Dataset
    # train_dataset = UCF101Dataset(root_dir, classes['train'], 6000, n_way=5, k_shot=5, log_path=f'{workdir}/train_data.txt')
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    # # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # Model
    # nseg = 4
    # model = C3DModel(nseg) 

    # # Train
    # ## iterate through episode batch
    # ## each data idx correspond to one episode
    # ## each data batch correspond to a batch of episodes
    
    # for id, data in enumerate(train_loader):
    #     data = to_gpu(data, device)
    #     predict_distance = model(data['support_set'], data['support_labels'], data['query_set'])     
    #     loss = train_loss(predict_distance, data['support_labels'], data['query_labels'])
    #     loss.backward()
    #     optimizer.step()

    #     if id % eval_iter == 0:
    #         eval(model, eval_loader)
    #         model.train()
    
    # # testing
    # ckpt = torch.load('best_model.pt')
    # model.load_state_dict(ckpt['state_dict'])
    # accuracy_list = []
    # for id, data in enumerate(test_loader):
    #     data = to_gpu(data, device)
    #     predict_distance = model(data['support_set'], data['support_labels'], data['query_set'])
    #     predict_labels = get_predict_labels(predict_distance, data['support_labels'])
    #     acc = accuracy(predict_labels, data['query_labels'])
    #     accuracy_list += acc
    
    # mean_accuracy = sum(accuracy_list)/len(accuracy_list)
    # confidence_interval = mean_confidence_interval(accuracy_list)
    # logger.info(f'Accuracy: {mean_accuracy} +- {confidence_interval}')



    # # must support logging, checkpointing
    # # test_dataset = UCF101Dataset()
    # # test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # # with torch.no_grad():
    # #     model.eval()
    # #     for data in train_loader:
    # #         predict_query_labels = model(data['support_set'], data['support_labels'], data['query_set'])
    # #         loss = train_loss(data['query_labels'], predict_query_labels)



if __name__ == '__main__':
    main()
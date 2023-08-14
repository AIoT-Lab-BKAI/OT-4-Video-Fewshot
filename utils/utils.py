import numpy as np
import scipy
from torch.utils.data import DataLoader
from easyfsl.samplers import TaskSampler
import os
import pytorch_lightning as pl
import yaml
import torch
import datetime
from qpth.qp import QPFunction
import torch.nn.functional as F

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

def get_episodic_dataloader(dataset, n_way, n_shot, n_query, n_tasks, num_workers=4):
    sampler = TaskSampler(dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True, collate_fn=sampler.episodic_collate_fn)
    return dataloader

def get_wandb_logger(logger_cfg):
    with open('secrets.yaml', 'r') as f:
        secrets = yaml.safe_load(f)
    os.environ["WANDB_API_KEY"] = secrets['wandb_api_key']
    logger = pl.loggers.WandbLogger(**logger_cfg)
    return logger

def get_checkpoint_callback(ckpt_cfg):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        **ckpt_cfg
    )
    return checkpoint_callback

def accuracy(logits, labels, calc_mean=True):
    labels = torch.flatten(labels)
    logits = logits.view(-1, logits.shape[-1])
    
    correct = torch.argmax(logits, 1) == labels
    if calc_mean:
        return torch.mean(correct.float())
    else:
        return correct.float()
    
def get_workdir(root_dir):
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    # Convert the current date and time to a string
    formatted_datetime = current_datetime.strftime('%m_%d_%H_%M')
    # Print the formatted datetime string
    path = os.path.join(root_dir, formatted_datetime)
    os.makedirs(path, exist_ok=True)
    return path

def emd_inference_qpth(distance_matrix, weight1, weight2, form='QP', l2_strength=0.0001):
    """
    to use the QP solver QPTH to derive EMD (LP problem),
    one can transform the LP problem to QP,
    or omit the QP term by multiplying it with a small value,i.e. l2_strngth.
    :param distance_matrix: nbatch * element_number * element_number
    :param weight1: nbatch  * weight_number
    :param weight2: nbatch  * weight_number
    :return:
    emd distance: nbatch*1
    flow : nbatch * weight_number *weight_number

    """


    weight1 = (weight1 * weight1.shape[-1]) / weight1.sum(1).unsqueeze(1)
    weight2 = (weight2 * weight2.shape[-1]) / weight2.sum(1).unsqueeze(1)

    nbatch = distance_matrix.shape[0]
    nelement_distmatrix = distance_matrix.shape[1] * distance_matrix.shape[2]
    nelement_weight1 = weight1.shape[1]
    nelement_weight2 = weight2.shape[1]

    Q_1 = distance_matrix.view(-1, 1, nelement_distmatrix).double()
    dev = distance_matrix.device
    if form == 'QP':
        # version: QTQ
        Q = torch.bmm(Q_1.transpose(2, 1), Q_1).double().to(dev) + 1e-4 * torch.eye(
            nelement_distmatrix).double().to(dev).unsqueeze(0).repeat(nbatch, 1, 1)  # 0.00001 *
        p = torch.zeros(nbatch, nelement_distmatrix).double().to(dev)
    elif form == 'L2':
        # version: regularizer
        Q = (l2_strength * torch.eye(nelement_distmatrix).double()).to(dev).unsqueeze(0).repeat(nbatch, 1, 1)
        p = distance_matrix.view(nbatch, nelement_distmatrix).double()
    else:
        raise ValueError('Unkown form')

    h_1 = torch.zeros(nbatch, nelement_distmatrix).double().to(dev)
    h_2 = torch.cat([weight1, weight2], 1).double()
    h = torch.cat((h_1, h_2), 1)

    G_1 = -torch.eye(nelement_distmatrix).double().to(dev).unsqueeze(0).repeat(nbatch, 1, 1)
    G_2 = torch.zeros([nbatch, nelement_weight1 + nelement_weight2, nelement_distmatrix]).double().to(dev)
    # sum_j(xij) = si
    for i in range(nelement_weight1):
        G_2[:, i, nelement_weight2 * i:nelement_weight2 * (i + 1)] = 1
    # sum_i(xij) = dj
    for j in range(nelement_weight2):
        G_2[:, nelement_weight1 + j, j::nelement_weight2] = 1
    #xij>=0, sum_j(xij) <= si,sum_i(xij) <= dj, sum_ij(x_ij) = min(sum(si), sum(dj))
    G = torch.cat((G_1, G_2), 1)
    A = torch.ones(nbatch, 1, nelement_distmatrix).double().to(dev)
    b = torch.min(torch.sum(weight1, 1), torch.sum(weight2, 1)).unsqueeze(1).double()
    flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)

    emd_score = torch.sum((1 - Q_1).squeeze() * flow, 1)
    return emd_score, flow.view(-1, nelement_weight1, nelement_weight2)

def cosine_similarity(set1, set2):
    set1_norm = torch.linalg.norm(set1, dim=1, keepdim=True)
    set1_normalized = set1 / set1_norm

    set2_norm = torch.linalg.norm(set2, dim=1, keepdim=True)
    set2_normalized = set2 / set2_norm

    # Compute cosine similarity
    cosine_sim= torch.mm(set1_normalized, set2_normalized.T)
    return cosine_sim

def criterion(pred_logits, target_labels, prefix=''):
    res = {}
    res[f'{prefix}loss'] = F.cross_entropy(pred_logits, target_labels)
    res[f'{prefix}acc'] = accuracy(pred_logits, target_labels)
    return res

def get_dataloader(dataloader_cfg):
    dataloader = {}
    for name, cfg in dataloader_cfg.items():
        dataset = cfg['dataset']
        dataloader[name] = get_episodic_dataloader(dataset, **cfg['sampler'])
    return dataloader

class Registry(object):
    def __init__(self):
        self._obj_map = {}

    def register(self, obj_type):
        def decorator(obj_cls):
            self._obj_map[obj_type] = obj_cls
            return obj_cls
        return decorator

    def get(self, obj_type):
        if obj_type not in self._obj_map:
            raise KeyError(f'Unkown type {obj_type}')
        return self._obj_map[obj_type]
    def register(self, obj_type, obj):
        self._obj_map[obj_type] = obj

def ot_distance(cost, dist1, dist2):
    return emd_inference_qpth(cost, dist1, dist2)

def get_marginal_distribution(n_batch, n_dim, device='cpu'):
    dist = torch.ones((n_batch, n_dim), device=device) / n_dim
    dist.requires_grad = False
    return dist
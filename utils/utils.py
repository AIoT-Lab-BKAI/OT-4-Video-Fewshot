import numpy as np
import scipy
from torch.utils.data import Dataset, DataLoader
from easyfsl.samplers import TaskSampler

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
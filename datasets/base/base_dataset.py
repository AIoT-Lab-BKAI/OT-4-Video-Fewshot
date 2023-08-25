from abc import abstractmethod
from torch.utils.data import Dataset
import os
import glob
import json

from easyfsl.samplers import TaskSampler


class BaseVideoDataset(Dataset):
    def __init__(self, root_dir, split_path, class_id_path, train=False):
        with open(split_path, 'r') as f:
            paths = f.readlines()
        
        paths = [path.strip() for path in paths]
        self.labels = [path.split('/')[0] for path in paths]
        video_paths = [os.path.join(root_dir, path) for path in paths]
        self.paths = [glob.glob(os.path.join(video_path, '*.jpg')) for video_path in video_paths]
        self.paths = [sorted(path) for path in self.paths]
        
        self.class_id = json.load(open(class_id_path, 'r'))
        self.labels = [self.class_id[label] for label in self.labels]

        self.train = train
    
    def __len__(self):
        return len(self.labels)
    
    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    def get_labels(self):
        return self.labels

class Base_Fewshot_Dataset(Dataset):
    def __init__(self, dataset, n_tasks, n_way, n_shot, n_query):
        super().__init__()
        self.dataset = dataset
        self.sampler = TaskSampler(self.dataset, n_tasks=n_tasks, n_way=n_way, n_shot=n_shot, n_query=n_query)
        self.iter = iter(self.sampler)
        self.n_tasks= n_tasks
    
    def __len__(self):
        return self.n_tasks
    
    def __getitem__(self, idx):
        try:
            ids = next(self.iter)
        except StopIteration:
            self.iter = iter(self.sampler)
            ids = next(self.iter)
        data = []
        for id in ids:
            data.append(self.dataset[id])
        data = self.sampler.episodic_collate_fn(data)
        res = dict(
            support_set = data[0],
            support_labels = data[1],
            query_set = data[2],
            query_labels = data[3]
        )
        return res
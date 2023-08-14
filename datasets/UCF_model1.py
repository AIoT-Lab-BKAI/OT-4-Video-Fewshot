from datasets.UCF101 import UCF101
import torchvision.transforms as transforms
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from easyfsl.samplers import TaskSampler
import configs.m1 as CFG
def sample_frames(frames, seq_len):
    while len(frames) < seq_len:
        frames += frames
    return random.sample(frames, seq_len)

class UCF101_M1(UCF101):
    def __init__(self, root_dir, split_path, classes, seq_len):
        super().__init__(root_dir, split_path, classes)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        self.seq_len = seq_len
    
    def __getitem__(self, idx):       
        frames = sample_frames(self.paths[idx], self.seq_len)
        frames = [Image.open(frame) for frame in frames]
        frames = [self.transforms(frame) for frame in frames]
        frames = torch.stack(frames, dim=0)

        return frames, torch.tensor(self.class_id[self.labels[idx]], dtype=torch.int32)
# a wrapper 
class FS_DS(Dataset):
    def __init__(self, dataset, n_tasks, n_way, n_shot, n_query):
        super().__init__()
        self.dataset = dataset
        self.sampler = TaskSampler(self.dataset, n_tasks=n_tasks, n_way=n_way, n_shot=n_shot, n_query=n_query)
        self.iter = iter(self.sampler)
        self.n_tasks= n_tasks
    
    def __len__(self):
        return self.n_tasks
    
    def __getitem__(self, idx):
        ids = next(self.iter)
        data = []
        for id in ids:
            data.append(self.dataset[id])
        data = self.sampler.episodic_collate_fn(data)
        return data[:4]

if __name__ == '__main__':
    cfg = CFG.dataloader['train']
    dataset_cfg = cfg['dataset']
    sampler_cfg = cfg['sampler']
    dataset = UCF101_m1(dataset_cfg, sampler_cfg)
    dataloader = DataLoader(dataset, batch_size=1)
    for batch in dataloader:
        import pdb; pdb.set_trace()

    

        
        

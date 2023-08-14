import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
from easyfsl.samplers import TaskSampler
import torchvision.transforms as transforms
import glob
from abc import abstractmethod

def sample_frames(frames, nseg, seglen):
    # get 4 segment of videos with 16 frames each
    total_len = nseg * seglen
    idx_list = list(range(len(frames)))
    while len(idx_list) < total_len:
        idx_list += idx_list
    
    start_idx = np.random.randint(len(idx_list) - seglen + 1, size=nseg)
    
    segments = []
    for i in range(nseg):
        x = torch.stack([frames[j] for j in idx_list[start_idx[i]:start_idx[i]+seglen]], dim=0)
        segments.append(x)
    segments = torch.stack(segments, dim=0)
    
    return segments

class UCF101(Dataset):
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
        pass

    def get_labels(self):
        return self.labels

class UCF101_C3D(Dataset):
    # all class
    def __init__(self, args):
        self.args = args

        data = []
        for cls in args.classes:
            cls_dir = os.path.join(args.root_dir, cls)
            data += [{"label": args.class_id[cls], "path": glob.glob(os.path.join(cls_dir, img, '*.jpg'))} for img in os.listdir(cls_dir)]
        self.data = data
        self.get_labels = lambda: [d['label'] for d in data]

        self.transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = [Image.open(path) for path in self.data[idx]['path']]
        frames = [self.transform(frame) for frame in frames]
        
        frames = sample_frames(frames, self.args.n_seg, self.args.seglen)        
        
        return frames, torch.tensor(self.data[idx]['label'], dtype=torch.int)

if __name__ == '__main__':

    classes = json.load(open('splits/val.json'))
    class_id = json.load(open('splits/classes.json'))
    args = edict(
        root_dir = 'data/ucf101',
        classes = classes,
        class_id = class_id,
        img_size = 112,
        n_seg = 4,
        seglen = 16,
    )
    dataset = UCF101_C3D(args)
    x, y = dataset[0]
    sampler = TaskSampler(dataset, n_way=5, n_shot=1, n_query=5, n_tasks=10)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=12, pin_memory=True, collate_fn=sampler.episodic_collate_fn)
    (
    example_support_images,
    example_support_labels,
    example_query_images,
    example_query_labels,
    example_class_ids,
    ) = next(iter(loader))
    import pdb; pdb.set_trace()
    




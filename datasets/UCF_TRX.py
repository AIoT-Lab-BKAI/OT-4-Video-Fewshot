import os
import random
import PIL
from torch.utils.data import Dataset, DataLoader
from easyfsl.samplers import TaskSampler
import torch
import torchvision.transforms as transforms
import numpy as np
import glob
from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, RandomHorizontalFlip, CenterCrop, TenCrop
from utils.utils import get_episodic_dataloader
from easydict import EasyDict as edict
class UCF_TRX(Dataset):
    def __init__(self, args):
        super(UCF_TRX, self).__init__()
        args = edict(args)
        self.args = args

        self.data_dir = args.data_dir
        self.seq_len = args.seq_len
        self.img_size = args.img_size
        self.train = args.train
        self.data_path = args.data_path
        
        with open(self.data_path, 'r') as f:
            self.data = f.readlines()
        self.data = [x.strip() for x in self.data]
        
        self.classes = [path.split('/')[0] for path in self.data]
        self.unique_classes = list(set(self.classes))
        sorted(self.unique_classes)
        self.class_id = {x:i for i, x in enumerate(self.unique_classes)}

        self.data = [os.path.join(self.data_dir, x) for x in self.data]
        self.data = [list(glob.glob(os.path.join(x, '*.jpg'))) for x in self.data]
        
        self.setup_transforms()
    
    def setup_transforms(self):
        video_transforms = []
        if self.img_size == 84:
            video_transforms.append(Resize(96))
        elif self.img_size == 224:
            video_transforms.append(Resize(256))
        else:
            print("img size transforms not setup")
            exit(1)
        if not self.train:
            video_transforms.append(CenterCrop(self.img_size))
        else:
            video_transforms.append(RandomHorizontalFlip())
            video_transforms.append(RandomCrop(self.img_size))

        self.video_transforms = Compose(video_transforms)
        self.tensor_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)
    
    def sample_frames(self, n_frames):
        if n_frames == self.seq_len:
            idxs = [int(f) for f in range(n_frames)]
        else:
            if self.train:
                excess_frames = n_frames - self.seq_len
                excess_pad = int(min(5, excess_frames / 2))
                if excess_pad < 1:
                    start = 0
                    end = n_frames - 1
                else:
                    start = random.randint(0, excess_pad)
                    end = random.randint(n_frames-1 -excess_pad, n_frames-1)
            else:
                start = 1
                end = n_frames - 2
    
            if end - start < self.seq_len:
                end = n_frames - 1
                start = 0
            else:
                pass
    
            idx_f = np.linspace(start, end, num=self.seq_len)
            idxs = [int(f) for f in idx_f]
            
            if self.seq_len == 1:
                idxs = [random.randint(start, end-1)]
        return idxs
    def __getitem__(self, idx):
        res = []
        for i in idx:
            res.append(self.__getitem1__(i))
        return res

    def __getitem1__(self, idx):
        frames_id = self.sample_frames(len(self.data[idx]))
        frames = [PIL.Image.open(self.data[idx][i]) for i in frames_id]
        frames = self.video_transforms(frames)
        frames = [self.tensor_transform(f) for f in frames]
        frames = torch.stack(frames, dim=0)

        return frames, torch.tensor(self.class_id[self.classes[idx]])
    
    def get_labels(self):
        return self.classes

if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict(data_path = 'splits/ucf_ARN/trainlist03.txt',
                data_dir='data/ucf101',
                img_size=224,
                seq_len=8,
                train=True,
                )
    dataset = UCF_TRX(args)
    dataloader = get_episodic_dataloader(dataset, 5, 5, 5, 10)
    for task in dataloader:
        import pdb; pdb.set_trace()
    

import random
import PIL
from datasets.UCF101 import UCF101
import torch
import torchvision.transforms as transforms
import numpy as np
from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, RandomHorizontalFlip, CenterCrop, TenCrop
from datasets.base.base_dataset import BaseVideoDataset
from datasets.base.builder import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Ucf101(BaseVideoDataset):
    def __init__(self, root_dir, split_path, class_id_path, img_size, seq_len, train=False):
        super(Ucf101, self).__init__(root_dir, split_path, class_id_path, train)
        self.img_size = img_size
        self.seq_len = seq_len
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
        self.tensor_trasforms = transforms.ToTensor()
    
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
        data = self.paths[idx]
        frames_id = self.sample_frames(len(data))
        frames = [PIL.Image.open(data[i]) for i in frames_id]
        
        frames = self.video_transforms(frames)
        frames = [self.tensor_trasforms(frame) for frame in frames]
        frames = torch.stack(frames, dim=0)

        labels = torch.tensor(self.labels[idx])

        return frames, labels

if __name__ == '__main__':
    pass

    

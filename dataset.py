import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
from easyfsl.samplers import TaskSampler

class UCF101Dataset(Dataset):
    CLASS_IDS = {cls:i for i, cls in enumerate(json.load(open('ucf101/classes.json', 'r')))}
    def __init__(self, root_dir, classes, nseg=4, seglen=16):
        self.nseg = nseg
        self.seglen = seglen
        self.normalize_offsets = [90.0, 98.0, 102.0]
        self.crop_size = 112
        # collect data instaces
        ## image dimension is width, height: 171, 128
        data = []
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            data += [{"label": UCF101Dataset.CLASS_IDS[cls], "path": os.path.join(cls_dir, img)} for img in os.listdir(cls_dir)]
        
        self.data = data
        self.get_labels = lambda: [d['label'] for d in data]

    def crop_frames(self, frames):
        crop_size = self.crop_size
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(frames.shape[2] - crop_size)
        width_index = np.random.randint(frames.shape[3] - crop_size)
        frames = frames[:, :, height_index:height_index + crop_size, width_index:width_index + crop_size, :]
    
        return frames
    
    def load_frames(self, frame_dir):
        frame_paths = sorted([os.path.join(frame_dir, img) for img in os.listdir(frame_dir)])
        frames = []
        for frame_path in frame_paths:
            frame = np.array(Image.open(frame_path).convert('RGB')).astype(np.float32)
            frames.append(frame)
        return frames

    def sample_frames(self, frames):
        # get 4 segment of videos with 16 frames each
        nseg = self.nseg
        seglen = self.seglen
        total_seglen = nseg * seglen
        while len(frames) < total_seglen:
            frames += frames
        
        start_idx = np.random.randint(len(frames) - total_seglen + 1)
        segments = [np.stack(frames[start_idx + i*seglen:start_idx + (i+1)*seglen]) for i in range(nseg)]
        segments = np.stack(segments)
        return segments
        
    def normalize(self, frames):
        offsets = self.normalize_offsets
        for i, offset in enumerate(offsets):
            frames[..., i] -= offset
        frames /= 255.0
        return frames
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = self.load_frames(self.data[idx]['path'])
        frames = self.sample_frames(frames)
        frames = self.crop_frames(frames)
        frames = self.normalize(frames)
        frames = frames.transpose(0, 4, 1, 2, 3)
        frames = torch.from_numpy(frames)
        return frames, torch.tensor(self.data[idx]['label'], dtype=torch.int)


if __name__ == '__main__':

    classes = json.load(open('ucf101/val.json'))    
    dataset = UCF101Dataset('data/ucf101', classes)
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
    




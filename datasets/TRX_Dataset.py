import os
import random
import PIL
from torch.utils.data import Dataset, DataLoader
from easyfsl.samplers import TaskSampler
import torch
import torchvision.transforms as transforms


def get_episodic_dataloader(dataset, n_way, n_shot, n_query, n_tasks, num_workers=4):
    sampler = TaskSampler(dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True, collate_fn=sampler.episodic_collate_fn)
    return dataloader

class TRX_Dataset(Dataset):
    def __init__(self, data_dir, class_id, classes, num_frames):
        super(TRX_Dataset, self).__init__()
        self.data = []
        self.labels = []
        for cls in classes:
            cls_dir = os.path.join(data_dir, cls)
            cls_instances = os.listdir(cls_dir)
            # sample num_frames for each video
            for instance in cls_instances:
                instance_path = os.path.join(cls_dir, instance)
                frame_paths = os.listdir(instance_path)
                frame_paths = [os.path.join(instance_path, frame) for frame in frame_paths]
                frame_paths = random.sample(frame_paths, num_frames)
                frame_paths = sorted(frame_paths)

                self.data.append(frame_paths)
                self.labels.append(class_id[cls])
        
        self.img_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frames = []
        for frame in self.data[idx]:
            image = PIL.Image.open(frame)
            image = self.img_transform(image)
            frames.append(image)
        frames = torch.stack(frames, dim=0)
        return frames, torch.tensor(self.labels[idx])
    
    def get_labels(self):
        return self.labels

if __name__ == '__main__':
    import json
    classes = json.load(open('config/train.json'))
    class_id = json.load(open('config/classes.json'))
    data_dir = 'data/ucf101'
    num_frames = 16
    dataset = TRX_Dataset(classes, class_id, data_dir, num_frames)
    x, y = dataset[0]
    dataloader = get_episodic_dataloader(dataset, 5, 1, 5, 10)
    for x1, x2, x3, x4, x5 in dataloader:
        import pdb; pdb.set_trace()

import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
class UCF101Dataset(Dataset):
    CLASS_IDS = {cls:i for i, cls in enumerate(json.load(open('ucf101/classes.json', 'r')))}
    def __init__(self, root_dir, classes, num_iter, n_way=5, k_shot=1, nseg=4, seglen=16, log_path=None, iter_data_path=None):
        self.n_segment = nseg
        self.segment_len = seglen
        self.normalize_offsets = [90.0, 98.0, 102.0]
        self.crop_size = 112
        # collect data instaces for each class
        ## image dimension is width, height: 171, 128
        assert isinstance(classes, list), "classes arg must be a list of class name"
        assert len(classes) >= n_way, f"number of classes, currently {len(classes)}, must be greater than or equal number of shot, currently {n_way}"
        self.data = {}
        self.cls_id = {}
        self.n_way = n_way
        self.k_shot = k_shot
        self.classes = classes
        for cls in classes:
            cls_data_dir = f'{root_dir}/{cls}'
            data_paths = os.listdir(cls_data_dir)
            assert len(data_paths) > k_shot, f"number of data instances for class {cls}, currently {len(data_paths)}, must be greater than k_shot, currently {k_shot}"
            data_paths = [os.path.join(cls_data_dir, path) for path in data_paths]
            self.data[cls] = data_paths
        
        # prepare data for each iteration
        self.log_path = log_path
        self.num_iter = num_iter
        if iter_data_path is None:
            ## support labels and query labels are the same
            assert num_iter > 0, "number of iteration must be greater than 0"
            
            self.sample_data()
            
            ## log iteration data
            if log_path is not None:
                logging_datas = [data['labels'] + data['support_set'] + data['query_set'] for data in self.iter_data]
                logging_datas = [','.join(data) for data in logging_datas]
                with open(log_path, 'w') as f:
                    for data in logging_datas:
                        f.write(data + '\n')
        else:
            with open(iter_data_path, 'r') as f:
                iter_data = f.readlines()
            iter_data = [data.strip().split(',') for data in iter_data]
            self.iter_data = []
            for data in iter_data:
                self.iter_data.append({
                    "labels": data[:n_way],
                    "support_set": data[n_way:n_way*k_shot],
                    "query_set": data[n_way*k_shot:]
                })
   
    def sample_data(self):
        self.iter_data = []
        for i in range(self.num_iter):
            sp_classes = random.sample(self.classes, self.n_way)
            sp_datas = []
            query_datas = []
            for cls in sp_classes:
                data = random.sample(self.data[cls], self.k_shot+1)
                sp_datas += data[:self.k_shot]
                query_datas += data[self.k_shot:]
            self.iter_data.append({'labels':sp_classes, "support_set":sp_datas, "query_set":query_datas}) 

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
            frame = np.array(Image.open(frame_path)).astype(np.float32)
            frames.append(frame)
        return frames

    def sample_frames(self, frames):
        # get 4 segment of videos with 16 frames each
        n_segment = self.n_segment
        segment_len = self.segment_len
        while len(frames) < n_segment * segment_len:
            frames += frames
        start_idx = np.random.randint(len(frames) - n_segment * segment_len + 1)
        res = np.stack([np.array(frames[i:i+segment_len]) for i in range(start_idx, start_idx+n_segment*segment_len, segment_len)])
        return res
        
    def normalize(self, frames):
        offsets = self.normalize_offsets
        for i, offset in enumerate(offsets):
            frames[..., i] -= offset
        return frames
    
    def __len__(self):
        return len(self.iter_data)

    def __getitem__(self, idx):
        labels = self.iter_data[idx]['labels']
        support_data_paths = self.iter_data[idx]['support_set']
        query_data_paths = self.iter_data[idx]['query_set']

        datas = []
        for path in support_data_paths + query_data_paths:
            data = self.load_frames(path)
            data = self.sample_frames(data)
            data = self.crop_frames(data)
            data = self.normalize(data)
            datas.append(data)
        
        support_set = np.stack(datas[:self.n_way*self.k_shot])
        support_set = support_set.reshape(self.n_way, self.k_shot, *support_set.shape[1:])
        support_set = support_set.transpose(0, 1, 2, 6, 3, 4, 5)

        query_set = np.stack(datas[self.n_way*self.k_shot:])
        query_set = query_set.reshape(self.n_way, self.k_shot, *query_set.shape[1:])
        query_set = query_set.transpose(0, 1, 2, 6, 3, 4, 5)


        labels = np.array([UCF101Dataset.CLASS_IDS[label] for label in labels])
        
        data = {
            "labels": labels,
            "support_set": support_set,
            "query_set": query_set
        }
        return data

if __name__ == '__main__':

    classes = json.load(open('ucf101/val.json'))    
    dataset = UCF101Dataset('data/ucf101', classes, 10, log_path='test.log')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for data in dataloader:
        import pdb; pdb.set_trace()
    




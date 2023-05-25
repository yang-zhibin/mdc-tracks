import torch 
import h5py
import pandas as pd
import numpy as np

class MdcTrackDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.event_ids = []
        self.event_data = []
        
        with h5py.File(self.file_path, 'r') as file:
            for event_id in file.keys():
                self.event_ids.append(event_id)
                self.event_data.append(file[event_id])

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx):
        event = self.event_data[idx]
        hit_feature = torch.Tensor(event['hit_feature'][()])
        hit_label = torch.Tensor(event['hit_label'][()])
        track_label = torch.Tensor(event['track_label'][()])
        return hit_feature, hit_label, track_label
    
class HitDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, split='train', split_ratio=[0.8, 0.1, 0.1], seed=0):
        super(HitDataset, self).__init__()
        assert split in ['train', 'val', 'test']
        assert sum(split_ratio) == 1.0

        self.split = split
        self.split_ratio = split_ratio
        self.file_path = file_path

        self.h5file = h5py.File(self.file_path, 'r')
        self.events = list(self.h5file.keys())
        self.num_events = len(self.events)

        if self.split == 'train':
            self.events = self.events[:int(self.num_events*self.split_ratio[0])]
        elif self.split == 'val':
            self.events = self.events[int(self.num_events*self.split_ratio[0]):int(self.num_events*(self.split_ratio[0]+self.split_ratio[1]))]
        elif self.split == 'test':
            self.events = self.events[int(self.num_events*(self.split_ratio[0]+self.split_ratio[1])):]

        self.event_ids = []
        self.event_data = []

        for event_name in self.events:
            event_group = self.h5file[event_name]
            hit_feat = np.array(event_group['hit_feature'])
            hit_label = np.array(event_group['hit_label'])
            track_label = np.array(event_group['track_label'])
            #event_info = np.array(event_group['event_info'])

            self.event_data.append((hit_feat, hit_label, track_label))
            self.event_ids.append(event_name)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        hit_feat, hit_label, track_label = self.event_data[index]
        event_id = self.event_ids[index]

        return {'event_id': event_id, 'hit_feature': hit_feat, 'hit_label': hit_label, 'track_label': track_label}
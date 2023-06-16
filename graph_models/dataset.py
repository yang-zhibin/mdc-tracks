import torch
import pandas as pd
import numpy as np
import h5py

# generate a pytorch dataset class

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

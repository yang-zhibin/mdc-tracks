import torch 
import h5py
import pandas as pd
import numpy as np
    
class HitDriftTimeDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, split='train', split_ratio=[0.5, 0.4, 0.1], seed=0):
        super(HitDriftTimeDataset, self).__init__()
        assert split in ['train', 'val', 'test']
        assert sum(split_ratio) == 1.0

        self.split = split
        self.split_ratio = split_ratio
        self.file_path = file_path

        self.data = pd.read_csv(self.file_path, nrows=1*10000)
        
        self.num_events = len(self.data)


        if self.split == 'train':
            self.data = self.data[:int(self.num_events*self.split_ratio[0])]
        elif self.split == 'val':
            self.fata = self.data[int(self.num_events*self.split_ratio[0]):int(self.num_events*(self.split_ratio[0]+self.split_ratio[1]))]
        elif self.split == 'test':
            self.data = self.data[int(self.num_events*(self.split_ratio[0]+self.split_ratio[1])):]

        self.rawDriftTime = self.data['rawDriftTime'] 
        self.driftDistance = self.data['driftDistance']


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rawDriftTime = self.rawDriftTime[index] /100.
        driftDistance = self.driftDistance[index] 

        return {'idx': index, 'rawDriftTime': rawDriftTime, 'driftDistance': driftDistance}
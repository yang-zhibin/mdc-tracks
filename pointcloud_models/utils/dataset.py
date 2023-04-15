from torch.utils.data import Dataset
import h5py
import pandas as pd
import numpy as np

class MDCTrackDataset(Dataset):
    def __init__(self, hdf5_file, key, transform=None):
        self.hdf5_file = hdf5_file
        self.key = key
        self.transform = transform
        self.data = pd.read_hdf(self.hdf5_file, self.key)
        self.data = self.data.values
        self.data = self.data.astype(np.float32)    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
    

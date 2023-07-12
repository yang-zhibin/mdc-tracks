import torch 
import h5py
import pandas as pd
import numpy as np

class MdcTrackDataset(torch.utils.data.Dataset):
    ''' hdf5 dataset for MDC tracking
    data are store event by event in hdf5 file, format is as follows:

    event_info = [event_type, run_id, event_id]
    hit_info = [gid]
    track_info = [particleProperty]

    hit_feature = [x, y, r, phi, rawDriftTime,rawDriftDistance] (unit[cm, cm, cm, rad, ns, cm])
    
    hit_label = [trackIndex, driftDistance] 
    track_label = [trackIndex, cx, cy, radius, charge] (unit[ , cm, cm, cm, ])
    '''

    def __init__(self, file_path):
        self.nmax_hits = 2048
        self.file_path = file_path
        self.event_ids = []
        self.hit_feature = []
        self.hit_label = []
        self.track_label = []
        
        with h5py.File(self.file_path, 'r') as file:
            for event_id in file.keys():
                event_data = file[event_id]
                
                self.event_ids.append(event_id)
                self.hit_feature.append( event_data['hit_feature'][:])
                self.hit_label.append( event_data['hit_label'][:])
                self.track_label.append(event_data['track_label'][:])
                
            
    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx):

        drop_noise = True

        event_id = self.event_ids[idx]
        hit_feature = np.zeros((self.nmax_hits, 3), dtype=np.float32)
        hit_label = np.zeros((self.nmax_hits, 2), dtype=np.float32)
        
        
        feature = np.array(self.hit_feature[idx].tolist())
        label = np.array(self.hit_label[idx].tolist())

        if (drop_noise):
            feature = feature[label[:,0]>0]
            label = label[label[:,0]>0]

        hit_feature[:len(feature)] = feature[:,[2,3,5]]
        hit_label[:len(label)] = label
        track_label = np.array(self.track_label[idx].tolist())
        
        
        return hit_feature, hit_label, track_label, event_id

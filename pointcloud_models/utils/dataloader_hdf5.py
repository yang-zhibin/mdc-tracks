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
        event = self.event_ids[idx]
        hit_feature = np.zeros((self.nmax_hits, 3), dtype=np.float32)
        hit_label = np.zeros((self.nmax_hits, 2), dtype=np.float32)
        
        
        x = self.hit_feature[idx][:,0]
        y = self.hit_feature[idx][:,1]
        r = self.hit_feature[idx][:,2]
        phi = self.hit_feature[idx][:,3]
        rawDriftTime = self.hit_feature[idx][:,4]
        rawDriftDistance = self.hit_feature[idx][:,5]

        hit_feature[:,0] = r
        hit_feature[:,1] = phi
        hit_feature[:,2] = rawDriftDistance

        trackid = self.hit_label[idx][:,0]
        driftDistance = self.hit_label[idx][:,1]

        hit_label[:,0] = trackid
        hit_label[:,1] = driftDistance

        track_label = self.track_label[idx]
        




        feature = np.array(self.hit_feature[idx].tolist())
        label = np.array(self.hit_label[idx].tolist())
        hit_feature[:len(feature)] = feature[:,2:]
        hit_label[:len(label)] = label
        
        track_label = np.array(self.track_label[idx].tolist())
        
        
        exp_key = self.event_ids[idx]
        trackid = self.track_label[idx]['trackid']
        
        
        
        
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
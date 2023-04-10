from torch.utils.data import Dataset
import h5py

class MDCTrackDataset(Dataset):
    def __init__(self, indir):
        self.indir = indir
        with h5py.File(self.indirectly， 'r') as file:
            self.dataset_len = len(file["dataset"])
    

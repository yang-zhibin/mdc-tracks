import h5py
import numpy as np

def read_hdf5():

    file_path = "D:\ihep\mdc-tracks\data\HDF5\single_anti-p-_0001.hdf5"
    h5file = h5py.File(file_path, 'r')
    events = list(h5file.keys())
    print (events)
    print(len(events))
    
    #for event_name in events:
    #    event_group = h5file[event_name]
    #    hit_feat = np.array(event_group['hit_feature'])
        
    #print(hit_feat)

def main():
    read_hdf5()
    


if __name__ == "__main__":
    main()

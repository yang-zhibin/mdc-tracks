import pandas as pd
import h5py
import os

# convert csv to hdf5 file
# generate a function to convert csv to hdf5 file   
def csv_to_hdf5(csv_file, hdf5_file, key):
    csv_data = pd.read_csv(csv_file)
    csv_data.to_hdf(hdf5_file, key, mode='w', format='table')



def main():
    rawData_dir = "./data/rawData/pipijpsi"
    out_dir = "./data/HDF5/pipijpsi"
    wirePos_file = "./data/preprocess/MdcWirePosition.csv"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    
    
if __name__ == '__main__':
    main()
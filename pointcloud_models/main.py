import os
import argparse
import torch
from utils.util import IOStream
from utils.dataset import MdcTrackDataset
import yaml

def main(args):
    # load config file
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
                
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Based Track Reconstrcution')
    parser.add_argument('--no-cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='config file')
    args = parser.parse_args()
    
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    main(args)
import argparse
import torch
from utils.util import IOStream

def main(args):
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Based Track Reconstrcution')
    parser.add_argument('--no-cuda', type=bool, default=False, help='enables CUDA training')
    args = parser.parse_args()
    
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    main(args)
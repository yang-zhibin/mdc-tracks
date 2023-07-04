import os
import argparse
import torch
from utils.util import IOStream
from utils.dataset import MdcTrackDataset
import yaml
from models.pointnet2_fit import pointnet2_fit, pointnet2_fit_ssg
from utils.coustom_loss import fit_loss



def main(args):
    # load config file
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    train_dataset = MdcTrackDataset(config['dataset']['train_path'])
    val_dataset = MdcTrackDataset(config['dataset']['val_path'])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=0)
    
    model = pointnet2_fit(normal_channel=False)
    evaluator = fit_loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['train']['lr'])
                
    for epoch in range(config['train']['epochs']):
        for batch_idx, data in enumerate(train_loader):
           #print(data)
           hit_feature = data[0]
           hit_label = data[1]
           track_label = data[2]
           
           #hit_feature.permute(0,2,1)
           pred, trans_feat = model(hit_feature)
           loss = evaluator(pred, track_label)
           loss.backward()
           optimizer.step()
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Based Track Reconstrcution')
    parser.add_argument('--no-cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--config', type=str, default='pointcloud_models\configs\config.yaml', help='config file')
    args = parser.parse_args()
    
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    main(args)
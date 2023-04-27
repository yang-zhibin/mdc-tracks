import argparse
import torch
from utils.util import IOStream
import yaml
from utils.dataset import HitDriftTimeDataset
from models.simpleNN import DriftDistanceRegression, PolynomialRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main(args):
    # load config file
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
    dataset_train = HitDriftTimeDataset(config['dataset']['train_path'], split='train', split_ratio=[1, 0, 0])
    dataset_val = HitDriftTimeDataset(config['dataset']['val_path'], split='val')
    
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config['train']['batch_size'], shuffle=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=config['train']['batch_size'], shuffle=False, num_workers=2)
    
    model = DriftDistanceRegression()
    #model = PolynomialRegression()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['train']['lr'])
    
    train_loss_list= pd.DataFrame(columns=['epoch', 'loss'])
    val_loss_list = pd.DataFrame(columns=['epoch', 'loss'])
    for epoch in range(config['train']['epochs']):
        train_loss = 0 
        data_x = []
        data_y = []
        pred_y = []
        for batch_idx, data in enumerate(train_loader):
            x = data['rawDriftTime'].to(torch.float)
            y = data['driftDistance'].to(torch.float)
            
            optimizer.zero_grad()
            y_hat = model(x.unsqueeze(1))
            #print(y_hat)
            loss = torch.nn.MSELoss()
            l = loss(y_hat, y).float()
            l.backward()
            
            train_loss += l.item()
            
            optimizer.step()
            
            data_x.extend(x.detach().tolist())
            data_y.extend(y.detach().tolist())
            pred_y.extend(y_hat.detach().tolist())
            
            if batch_idx % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), l.item()))
        #print(pred_y)   
        if epoch%config['train']['val_interval'] == 0:
            val_loss = val(model, val_loader, epoch)
            val_loss_list = val_loss_list.append({'epoch': epoch, 'loss': val_loss}, ignore_index=True)
        
        train_loss /= len(train_loader.dataset)
        
        
        train_loss_list = train_loss_list.append({'epoch': epoch, 'loss': train_loss}, ignore_index=True)
        
        diplsay(data_x, data_y, pred_y, train_loss, epoch, 'train')
        print('Train set: Average loss: {:.4f}'.format(train_loss))
        
    fig = plt.figure(figsize=(20, 8))
    
    plt.plot(train_loss_list['epoch'], train_loss_list['loss'], label='train loss')
    plt.plot(val_loss_list['epoch'], val_loss_list['loss'], label='val loss')

    plt.legend()
    plt.xlabel('epoch')
    plt.title('loss')
    plt.savefig('driftTimeToDistance\plots\loss_vs_epoch.png')
    plt.close(fig)

    train_loss_list.to_csv('driftTimeToDistance\plots\\train_loss.csv')
    val_loss_list.to_csv('driftTimeToDistance\plots\\val_loss.csv')

def diplsay(data_x, data_y, y_pred, loss, epoch, split):
    fig = plt.figure(figsize=(20, 16))
    #print(data_x)
    data_x = [i*100 for i in data_x]
    #data_y = [i*13 for i in data_y]
    #y_pred = [i*13 for i in y_pred]
    
    axs1 = fig.add_subplot(111)
    axs1.scatter(data_x, data_y, alpha=0.5, label='ground truth')
    axs1.scatter(data_x, y_pred, alpha=0.8, label = 'prediction')
    plt.title('Ground Truth vs Prediction Epoch: {}'.format(epoch))
    plt.xlabel('rawDriftTime (ns)')
    plt.ylabel('driftDistance (mm)')
    plt.xlim(0, 1500)
    plt.ylim(0, 13)
    plt.text(0.01, 0.95, 'Average loss: {}'.format(loss), transform=axs1.transAxes)
    plt.legend()
    
    out_file = 'driftTimeToDistance\plots\{}_epoch{}.png'.format(split, epoch)
    plt.savefig(out_file)
    plt.close(fig)

def val(model, val_loader, epoch):
    val_loss = 0
    y_pred = []
    data_x = []
    data_y = []
    for batch_idx, data in enumerate(val_loader):
        x = data['rawDriftTime'].to(torch.float)
        y = data['driftDistance'].to(torch.float)
                  
        y_hat = model(x.unsqueeze(1))
        loss = torch.nn.MSELoss()
        l = loss(y_hat, y)
        l.backward()
        val_loss += l.item()
        
        data_x.extend(x.detach().tolist())
        data_y.extend(y.detach().tolist())
        y_pred.extend(y_hat.detach().tolist())
        
    val_loss /= len(val_loader.dataset)
    print('Val set: Average loss: {:.4f}'.format(val_loss))
    
    diplsay(data_x, data_y, y_pred, val_loss, epoch, 'val')
    
    return val_loss
    
        
        
        
   
                
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Based Track Reconstrcution')
    parser.add_argument('--no-cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--config', type=str, default='driftTimeToDistance\configs\config.yaml', help='config file')
    args = parser.parse_args()
    
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    main(args)
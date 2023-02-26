import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def draw_result(train_log, val_log, pdf):
    train_epoch = train_log['epoch']
    val_epoch = val_log['epoch']
    event_num = train_log.loc[0, 'event_num']
    train_log = train_log.drop(columns=['event_num', 'loss_sum', 'exists', 'param', 'circle_dist'])
    
    fig, (loss_axis, matric_axis) = plt.subplots(2, 1, figsize= [6.4, 9.6])
    for index, col in train_log.iteritems():
        if (index=='epoch'):
            continue
        if (index.find('loss')!=-1):
            loss_axis.plot(train_epoch, col, label=index)
        else:
            matric_axis.plot(train_epoch, col, label=index)
    loss_axis.set_title('Train Loss')
    loss_axis.set_xlabel('Epoch')
    loss_axis.set_ylabel('Loss')
    loss_axis.legend()
    
    matric_axis.set_title('Train Performance')
    matric_axis.set_xlabel('Epoch')
    matric_axis.set_ylabel('Performance')
    matric_axis.legend()
    pdf.savefig(fig)
    
    fig2, (loss_axis, matric_axis) = plt.subplots(2, 1, figsize= [6.4, 9.6])
    for index, col in val_log.iteritems():
        if (index=='epoch'):
            continue
        if (index.find('loss')!=-1):
            loss_axis.plot(val_epoch, col, label=index)
        else:
            matric_axis.plot(val_epoch, col, label=index)
    loss_axis.set_title('Val Loss')
    loss_axis.set_xlabel('Epoch')
    loss_axis.set_ylabel('Loss')
    loss_axis.legend()
    
    matric_axis.set_title('Val Performance')
    matric_axis.set_xlabel('Epoch')
    matric_axis.set_ylabel('Performance')
    matric_axis.legend()
    pdf.savefig(fig2)
    


def read_log(log_data):
    log_split = log_data.split(sep='Epoch#')
    col = ['epoch']
    train_log = pd.DataFrame(columns=col)
    val_log = pd.DataFrame(columns=col)

    for index in range (1, len(log_split)-1):
        #print(index)
        val_epoch = False
        epochs = log_split[index]
        one_epoch = epochs.split(sep='Train set')
        one_epoch = one_epoch[-1]
        #print(one_epoch.find('Validation set'))
        if (one_epoch.find('Validation set')==-1):
            train_set = one_epoch
            train_set = train_set.split(sep='Metrics:')
            train_loss = train_set[0]
            train_matric = train_set[1]
        else:
            one_epoch = one_epoch.split(sep='Validation set')
            train_set = one_epoch[0]
            train_set = train_set.split(sep='Metrics:')
            train_loss = train_set[0]
            train_matric = train_set[1]
            
            val_set = one_epoch[1]
            val_set = val_set.split(sep='Metrics:')
            val_loss = val_set[0]
            val_matric = val_set[1]
            
            val_loss = val_loss.split(sep='\n')[1]
            val_loss = val_loss.split(sep='Loss terms:')[1]
            val_matric = val_matric.split(sep='\n')[0]
            
            val_epoch = True
        
        train_loss = train_loss.split(sep='\n')[1]
        train_loss = train_loss.split(sep='Loss terms:')[1]
        train_matric = train_matric.split(sep='\n')[0]
        
        train_log.loc[len(train_log)] = index
        train_loss = train_loss.split(sep=',')
        for j in range(len(train_loss)):
            loss_item = train_loss[j].split(sep=':')
            loss_name = loss_item[0].lstrip()
            loss_value = float(loss_item[1].lstrip())
            train_log.loc[len(train_log)-1, loss_name] = loss_value
            
        train_matric = train_matric.split(sep=',')
        for j in range(len(train_matric)):
            matric_item = train_matric[j].split(sep=':')
            matric_name = matric_item[0].lstrip()
            matric_value = float(matric_item[1].lstrip())
            train_log.loc[len(train_log)-1, matric_name] = matric_value
            
        if (val_epoch):
            val_log.loc[len(val_log)] = index
            val_loss = val_loss.split(sep=',')
            for j in range(len(val_loss)):
                loss_item = val_loss[j].split(sep=':')
                loss_name = loss_item[0].lstrip()
                loss_value = float(loss_item[1].lstrip())
                val_log.loc[len(val_log)-1, loss_name] = loss_value
            
            val_matric = val_matric.split(sep=',')
            for j in range(len(val_matric)):
                matric_item = val_matric[j].split(sep=':')
                matric_name = matric_item[0].lstrip()
                matric_value = float(matric_item[1].lstrip())
                val_log.loc[len(val_log)-1, matric_name] = matric_value
                
    event_num = int(log_data[log_data.find('Train Epoch:') + len('Train Epoch: 1 ['):log_data.find('*(')])
    train_log['event_num'] = event_num
    
    return train_log, val_log

def compare_train_loss(train_log):
    print()

def main():
    log_path = './results/h2t_v1_2/network.log'
    output_pdf = log_path.replace('network.log', 'train_curves.pdf')
    train_csv = log_path.replace('network.log', 'train_log.csv')
    val_csv = log_path.replace('network.log', 'val_log.csv')
    with open(log_path, 'r') as f:
        log_data = f.read()

    train_log, val_log = read_log(log_data)
    train_log.to_csv(train_csv)
    val_log.to_csv(val_csv)

    with PdfPages(output_pdf) as pdf:
        draw_result(train_log, val_log, pdf)
        print('Save to pdf file',pdf)

    compare_train_loss(train_log)

if __name__ == '__main__':
    main()
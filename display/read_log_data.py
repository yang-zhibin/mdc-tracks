import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def draw_result(train_log, val_log, pdf):
    epoch = train_log['epoch']
    val_epoch = val_log['epoch']
    '''
    for index, col in train_log.iteritems():
        if (index == 'epoch'):
            continue

        plt.plot(epoch, col, 'o-')
        plt.xlabel('epoch')
        plt.ylabel('train '+index)
        plt.title('train '+index)
        pdf.savefig()
        plt.close()

        #val_col = val_log[index]
        #plt.plot(val_epoch, val_col, 'x')
        #plt.xlabel('epoch')
        #plt.ylabel('val ' + index)
        #plt.title('val ' + index)
        #pdf.savefig()
        #plt.close()
    '''

    train_loss = train_log.iloc[:, 1:4]
    tran_acc = train_log.iloc[:, 4:]

    #train_loss = train_loss.drop(train_loss[train_loss.loss_param > 5].index)
    train_loss_left = []
    train_acc_left = []


    train_loss.plot()
    plt.xlabel('epoch')
    plt.ylabel('loss, ')
    plt.title('train loss')
    pdf.savefig()
    plt.close()

    train_loss = train_loss.apply(lambda x: x / x[0])

    train_loss.plot()
    plt.xlabel('epoch')
    plt.ylabel('loss, devided by the initial value')
    plt.title('train loss')
    plt.ylim(0, 2)
    pdf.savefig()
    plt.close()

    tran_acc.plot()
    plt.xlabel('epoch')
    plt.ylabel('performance, %')
    plt.title('train performance')

    pdf.savefig()
    plt.close()


def read_log(log_data):
    log_split = log_data.split(sep='Epoch#')

    #col = ['epoch', 'loss_exist', 'loss_group', 'loss_param',
    #       'exists', 'group_acc']
    #col = ['epoch', 'loss_exist', 'loss_group', 'loss_param',
    #       'exists', 'group_oacc']
    #col = ['epoch', 'loss_exist', 'loss_group','loss_param', 'loss_hits_distance','loss_param_distance',
    #       'exists', 'group_oacc']
    col = ['epoch', 'loss_exist', 'loss_group', 'loss_param',
           'exists', 'group_oacc']
    train_log = pd.DataFrame(columns=col)
    val_log = pd.DataFrame(columns=col)

    for index in range(len(log_split) - 1):
        #print(index)
        i = log_split[index]
        if index > 0:
            train_split = i.split(sep='Train set')
            train_set = train_split[-1]
            val_split = train_set.split(sep='Validation set')
            train_data = val_split[0]
            train_log.loc[len(train_log)] = 0
            for data_name in col:

                if (data_name == 'epoch'):
                    train_log.loc[len(train_log) - 1, 'epoch'] = index
                    continue
                #print(data_name)
                data_name_index = train_data.rfind(data_name) + len(data_name) + 2
                data_name_digi = train_data[data_name_index:data_name_index + 6]
                train_log.loc[len(train_log) - 1, data_name] = float(data_name_digi)

            # loss_exist = train_data[loss_exist_index + len('loss_exist')+2:]
            if (len(val_split) > 1):
                val_log.loc[len(val_log)] = 0
                val_data = val_split[1]
                for data_name in col:
                    if (data_name == 'epoch'):
                        val_log.loc[len(val_log) - 1, 'epoch'] = index
                        continue
                    data_name_index = val_data.rfind(data_name) + len(data_name) + 2
                    data_name_digi = val_data[data_name_index:data_name_index + 6]
                    val_log.loc[len(val_log) - 1, data_name] = float(data_name_digi)

    return train_log, val_log

def compare_train_loss(train_log):
    print()

def main():
    log_path = 'E:\ihep\BESIII\hep_track\output_2\\h2t_polar\\h2t_polar_v3.log'
    with open(log_path, 'r') as f:
        log_data = f.read()

    train_log, val_log = read_log(log_data)

    output = 'h2t_polar_v3_results.pdf'
    with PdfPages(output) as pdf:
        draw_result(train_log, val_log, pdf)
        print('Save to pdf file',pdf)

    compare_train_loss(train_log)

if __name__ == '__main__':
    main()
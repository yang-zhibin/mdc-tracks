import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

def display_data(data, input_dir):
    fig = plt.figure(figsize=(10, 4))
    d = pd.DataFrame(columns=['d0', 'type'])
    d_truth = pd.DataFrame(columns=['d0', 'type'])

    d['d0'] = data['d'] * 100
    d['type'] = 'd0_pre'
    d_truth['d0'] = data['d_truth'] * 100
    d_truth['type'] = 'd0_truth'
    d0 = pd.concat([d, d_truth], ignore_index=True)

    axs1 = fig.add_subplot(121)
    axs1 = sns.histplot(data=d0, x='d0', hue='type')
    axs1.set(xlabel = 'd0 (cm)')

    axs3 = fig.add_subplot(122)
    axs3 = sns.histplot((data['d'] - data['d_truth'])*100,  stat='density')
    axs3.set(xlabel='Δd0 (cm)')
    x0, x1 = axs3.get_xlim()
    dd = data['d'] - data['d_truth']
    #dd = dd[(dd>-0.3)]
    mean = dd.mean() * 100
    std = dd.std() * 100
    print(std)
    x = np.linspace(x0, x1, num=100)
    gauss = stats.norm(loc=mean, scale=std)
    y = gauss.pdf(x)
    axs3.plot(x, y, color='red')
    title = "Δd0 distribution σ = %s μ = %s (total count:%s)"%(round(std, 3),round(mean,3), data.shape[0])
    plt.title(title)

    plt.savefig(input_dir+'/distance.png')
    plt.close(fig)

    pt= pd.DataFrame(columns=['pt', 'type'])
    pt_truth = pd.DataFrame(columns=['pt', 'type'])

    pt['pt'] = data['pt']
    pt['type'] = 'pt_pre'
    pt_truth['pt'] = data['pt_truth']
    pt_truth['type'] = 'pt_truth'
    pt = pd.concat([pt, pt_truth], ignore_index=True)

    fig = plt.figure(figsize=(10, 4))
    axs1 = fig.add_subplot(121)
    axs1 = sns.histplot(data=pt, x='pt', hue='type')
    axs1.set(xlabel='pt (GeV)')
    #axs2 = fig.add_subplot(122)
    #axs2 = sns.distplot(data['pt_truth'])
    #axs2.set(xlabel='pt_truth (GeV)')
    axs3 = fig.add_subplot(122)
    axs3 = sns.histplot(data['pt'] - data['pt_truth'], stat='density')
    axs3.set(xlabel='Δpt (GeV)')
    x0, x1 = axs3.get_xlim()
    dpt = data['pt'] - data['pt_truth']
    mean = dpt.mean()
    std = dpt.std()
    print(std)
    x = np.linspace(x0, x1, num=100)
    gauss = stats.norm(loc=mean, scale=std)
    y=gauss.pdf(x)
    axs3.plot(x, y, color='red')
    title = "pt distribution σ = %s μ = %s(total count:%s)"%(round(std, 3),round(mean,3), data.shape[0])
    plt.title(title)

    plt.savefig(input_dir+'/pt.png')
    plt.close(fig)


def display_eff(eff, input_dir):
    total_event = eff.shape[0]
    thread = [1, 0.99, 0.95, 0.9,0.8, 0.6, 0.5,0.4,0.3,0.2,0]
    thread.sort()
    pt_thread = pd.DataFrame(columns=['pt', 'thread'])
    for index in range(len(thread)):
        t=thread[index]
        temp = pd.DataFrame(columns=['pt', 'thread'])
        eff_temp = eff.drop(eff[eff.eff<=t].index)
        temp['pt'] = eff_temp['pt_truth']
        temp['thread'] = t
        pt_thread = pd.concat([pt_thread, temp],ignore_index=True)

    fig = plt.figure(figsize=(10, 4))
    axs1 = fig.add_subplot(121)
    axs1 = sns.histplot(eff,x='pt_truth',y='eff', bins=30, cbar=True, cmap='rocket_r')
    #axs1 = sns.kdeplot(eff['pt_truth'], label='density')
    axs1.set(xlabel='pt (GeV)', ylabel='hits eff', xlim=(0, None))
    title = "hit_eff distribution on Pt (total count:%s)" % (total_event)
    plt.title(title)
    #plt.legend()
    plt.savefig(input_dir+'/hits_eff.png')
    plt.close(fig)

    pt_thread=pt_thread.rename(columns={'thread':'threshold'})
    #fig = plt.figure(figsize=(10, 8))
    axs2 = fig.add_subplot(122)
    axs2 = sns.kdeplot(data=pt_thread, x='pt', hue='threshold')
    #, palette="ch:rot=-.25,hue=1,light=.75"
    axs2.set(xlabel='pt (GeV)')
    plt.title('Different hit_eff threshold on pt distribution')


    #val_count = pt_thread['thread'].value_counts()

    plt.savefig(input_dir+'/hit_eff_pt_cut_v9.png')
    plt.close(fig)



    x=[]
    y = []
    for index in range(len(thread)):
        t=thread[index]
        temp = eff.drop(eff[eff.eff < t].index)
        n1, b1, p1 = plt.hist(eff['pt_truth'], 10)
        n2, b2, p2 = plt.hist(temp['pt_truth'], 10)
        b = []
        for i in range(len(b1) - 1):
            b.append((b1[i] + b1[i + 1]) / 2.)
        n = n2 / n1

    plt.close()
    track_eff = eff.drop(eff[eff.eff < 0.5].index)

    fig = plt.figure(figsize=(20, 8))
    axs1 = fig.add_subplot(131)
    axs1 = sns.histplot(eff['pt_truth'], alpha=0.5, bins=10, color='yellow')
    axs1 = sns.histplot(track_eff['pt_truth'], alpha=0.5, bins=10, color='blue')
    axs1.set(xlabel='pt (GeV)')
    plt.legend(labels=['Gnd_num', 'Found_num'])
    title = "num of track found on Pt (total count:%s)" % (total_event)
    plt.title(title)

    fig = plt.figure(figsize=(20, 8))
    axs3 = fig.add_subplot(133)


    axs3= plt.scatter(x=x, y=y)

    #title = "num of track on Pt (total event:%s)" % (total_event)
    #plt.title(title)

    plt.savefig(input_dir+'/track_eff.png')
    plt.close(fig)


    gnd_track_num = eff.value_counts('event')
    pre_track_num = track_eff.value_counts('event')

    fig = plt.figure()
    axs1 = fig.add_subplot(121)
    axs1 = sns.histplot(gnd_track_num)
    axs1.set(xlabel='gnd num of track')
    axs2 = fig.add_subplot(122)
    axs2 = sns.histplot(pre_track_num)
    axs2.set(xlabel='prediction num of track')
    plt.savefig(input_dir+'/num_of_track.png')
    plt.close(fig)



def main():
    input_dir = './results/h2t_polar_distance_v2/prediction'
    match_track_file = input_dir + '/match_track_data.csv'
    eff_file = input_dir + '/track_eff_data.csv'
    data = pd.read_csv(match_track_file)
    eff_data = pd.read_csv(eff_file)
    eff_data = eff_data.drop(eff_data[eff_data.trackId_gnd == 0].index)

    display_data(data, input_dir)
    display_eff(eff_data, input_dir)

if __name__ == '__main__':
    main()
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

def process_event(event):
    event_gnd = event.groupby('trackId_gnd')

    hits = event
    hits['tp'] = 1
    purity_df = pd.DataFrame(columns=['trackId_pred', 'purity'])
    track_purity_sum = 0
    count_pred = 0
    event_pred =event.groupby('trackId_pred')
    purity_sample = pd.DataFrame(columns=['trackId_pred', 'match_gnd', 'total_hits', 'match_hits', 'purity'])
    match = pd.DataFrame(columns=['pred', 'gnd'])
    for track_id, track in event_pred:
        val_count = track['trackId_gnd'].value_counts()
        #if (val_count.sum()<6):
        #    continue
        purity = float(val_count.max()) / float(val_count.sum())
        if (purity > 0.5):
            match.loc[len(match.index)] = [int(track_id), int(val_count.idxmax())]
        else:
            count_pred += 1
            continue
        track_purity_sum += purity
        count_pred += 1
        purity_df.loc[len(purity_df.index)] = [track_id, round(purity)]
        purity_sample.loc[len(purity_sample.index)] = [int(track_id), int(val_count.idxmax()), val_count.sum(), val_count.max(), round(purity,3)]
        hits.loc[track[track.trackId_gnd == val_count.idxmax()].index, 'tp'] = 0

    track_purity = track_purity_sum / count_pred

    #hits = hits.drop(hits[hits.tp == 1].index)

    track_eff_sum = 0
    count_gnd = 0
    eff_df = pd.DataFrame(columns=['trackId_gnd', 'eff'])
    eff_sample = pd.DataFrame(columns=['trackId_gnd', 'match_pred', 'total_hits', 'match_hits', 'eff'])
    match_grouped = match.groupby('gnd')
    for track_id, track in event_gnd:
        val_count = track['trackId_pred'].value_counts()
        #if (val_count.sum()<6):
        #    continue
        sum = 0
        eff_match_pred = ''
        if (track_id not in match['gnd'].values):
            eff = 0
            match.loc[len(match.index)] = ['no match', track_id]
            eff_match_pred = 'No match'
        else:
            match_track = match_grouped.get_group(track_id)
            for id, tk in match_track.iterrows():
                sum +=val_count[tk['pred']]
                eff_match_pred = eff_match_pred + str(tk['pred']) +','
            eff = sum/val_count.sum()
        track_eff_sum += eff
        count_gnd += 1
        eff_df.loc[len(eff_df.index)] = [track_id, round(eff,3)]
        eff_sample.loc[len(eff_sample.index)] = [int(track_id), eff_match_pred, val_count.sum(), sum, round(eff,3)]


    track_eff = track_eff_sum / count_gnd

    match.sort_values(by='gnd')
    return track_eff, track_purity, eff_df, purity_df, eff_sample, purity_sample, hits, match


def draw_result(result,  pdf):

    draw_event = 200

    pred_grouped = result.groupby('event')

    event_eff_sum = 0
    event_purity_sum = 0
    count = 0
    wrong_count = 0
    no_match_track = 0
    sum_track = 0
    match_hit = 0
    sum_hit = 0
    for event_id, event in pred_grouped:
        if(len(event.index)<6):
            continue
        track_eff, track_purity, eff_df, purity_df, eff_sample, purity_sample, hits, match= process_event(event)
        event_eff_sum += track_eff
        event_purity_sum +=track_purity
        count += 1
        print(count)

        eff_sample = eff_sample.drop(eff_sample[eff_sample.trackId_gnd == 0].index)
        if (eff_sample.shape[0]<1):
            continue

        for id, tk in eff_sample.iterrows():
            if(tk['eff']>0.5):
                sum_track += 1
            else:
                no_match_track +=1
                sum_track +=1

        #if (0 not in eff_sample['eff'].values):
        #    sum_track += eff_sample.shape[0]
        #else:
         #   val_count = eff_sample['eff'].value_counts()
         #   no_match_track += val_count[0]
         #   print('0 count', val_count[ 0])
         #   sum_track += eff_sample.shape[0]

        match_hit += eff_sample['match_hits'].sum()
        sum_hit +=eff_sample['total_hits'].sum()
        # ((eff_sample.shape[0] + 1 != purity_sample.shape[0]) & (count < 200))
        if (count<200 ):
            print('drawing event:', count)
            fig = plt.figure(figsize=(20, 8))
            axs1 = fig.add_subplot(121)
            outterWall = plt.Circle((0, 0), 0.81, fill=False, alpha=0.2)
            innerWall = plt.Circle((0, 0), 0.06, fill=False, alpha=0.2)
            axs1.add_artist(outterWall)
            axs1.add_artist(innerWall)
            axs1.set_xlim((-0.82, 0.82))
            axs1.set_ylim((-0.82, 0.82))
            axs1.set_aspect(1)

            sc1 = axs1.scatter(event['x'], event['y'], marker='o',
                               c=event['trackId_gnd'], cmap=plt.cm.Set1_r, alpha=0.7)
            plt.colorbar(sc1, fraction=0.05)

            plt.title('event ' + str(int(event_id)) + ' Ground Truth')

            axs2 = fig.add_subplot(122)
            outterWall = plt.Circle((0, 0), 0.81, fill=False, alpha=0.2)
            innerWall = plt.Circle((0, 0), 0.06, fill=False, alpha=0.2)
            axs2.add_artist(outterWall)
            axs2.add_artist(innerWall)
            axs2.set_xlim((-0.82, 0.82))
            axs2.set_ylim((-0.82, 0.82))
            axs2.set_aspect(1)

            sc2 = axs2.scatter(event['x'], event['y'], marker='o',
                               c=event['trackId_pred'], cmap=plt.cm.Set1_r, alpha=0.5)
            plt.colorbar(sc2, fraction=0.05)
            axs2.scatter(hits['x'], hits['y'], marker='1', c=hits['trackId_pred'], cmap=plt.cm.Set2_r, alpha=hits['tp'])
            #axs2.table(cellText=eff_df.values, colLabels=eff_df.columns,colWidths=[0.1,0.1], loc='upper left')
            #axs2.table(cellText=purity_df.values, colLabels=purity_df.columns, colWidths=[0.1,0.1], loc='upper right')

            axs2.table(cellText=eff_sample.values, colLabels=eff_sample.columns,colWidths=[0.18,0.18,0.18,0.18,0.18], loc='upper left')
            axs2.table(cellText=purity_sample.values, colLabels=purity_sample.columns, colWidths=[0.18,0.18,0.18,0.18,0.18], loc='lower left')
            axs1.table(cellText=match.values, colLabels=match.columns, colWidths=[0.1, 0.1], loc='upper center')

            #if (len(eff_df.index)!=len(purity_df.index)):
            #    axs2.text(-0.4, -0.7, 'Quantity of prediction trackIndex is wrong!', c='red')
            plt.title('event ' + str(int(event_id+1)) + ' Prediction')
            pdf.savefig()
            plt.close()
        if (len(eff_df.index) != len(purity_df.index)):
            wrong_count += 1

        if(count>200):
            break
    eff = event_eff_sum / count
    purity = event_purity_sum / count
    epoch = result.loc[0, 'epoch']
    Qty_acc= 1.0-float(wrong_count)/float(count)

    track_eff = 1.0-no_match_track/sum_track
    hit_eff = match_hit / sum_hit
    result_df = pd.DataFrame(columns=['epoch', 'event_count', 'track_eff','hit_eff', 'hit purity'])
    result_df.loc[0] = [epoch, count, track_eff,hit_eff, purity]
    fig = plt.figure(figsize=(20, 8))
    plt.table(cellText=result_df.values, colLabels=result_df.columns, loc='center')
    plt.title('overall results')
    pdf.savefig()
    plt.close()
    print('eff:', eff, 'purity:', purity)

def match_track(event):
    pre_grouped = event.groupby('trackId_pred')
    gnd_grouped = event.groupby('trackId_gnd')

    pre_base = pd.DataFrame(columns=['trackId_pred', 'match_gnd', 'total_hits', 'match_hits', 'purity'])
    gnd_base = pd.DataFrame(columns=['trackId_gnd', 'match_pred', 'total_hits', 'match_hits', 'eff'])
    gnd_base_2 = pd.DataFrame(columns=['trackId_gnd', 'match_pred', 'total_hits', 'match_hits', 'eff'])

    for track_id, track in pre_grouped:
        gnd_count = track.value_counts('trackId_gnd')
        track_purity = float(gnd_count.max()) / float(gnd_count.sum())
        if (track_purity > 0.5):
            pre_base.loc[len(pre_base.index)] = [int(track_id), int(gnd_count.idxmax()), gnd_count.sum(), gnd_count.max(), round(track_purity,3)]
        else:
            pre_base.loc[len(pre_base.index)] = [int(track_id), -1, gnd_count.sum(), 0, 0]

    pre_base_grouped = pre_base.groupby('match_gnd')


    for track_id, track in gnd_grouped:
        if (track_id in pre_base['match_gnd'].values):
            match_pre = pre_base_grouped.get_group(track_id)
            pre_count = track.value_counts('trackId_pred')

            total_hit = pre_count.sum()
            match_hit = 0
            for id, tk in match_pre.iterrows():
                match_hit += pre_count[tk['trackId_pred']]

            track_eff = match_hit / total_hit

            gnd_base_2.loc[len(gnd_base.index)] = [int(track_id), pre_count.idxmax(), total_hit, match_hit, track_eff]
            if (track_eff>0.5):
                gnd_base.loc[len(gnd_base.index)] = [int(track_id), pre_count.idxmax(), total_hit, match_hit, track_eff]
            else:
                gnd_base.loc[len(gnd_base.index)] = [int(track_id), -1, total_hit, 0, 0]
        else:
            pre_count = track.value_counts('trackId_pred')
            total_hit = pre_count.sum()
            gnd_base.loc[len(gnd_base.index)] = [int(track_id), -1, total_hit, 0, 0]


    return gnd_base, pre_base, gnd_base_2
    print('test')

def cal_d(cx, cy, r):
    d = math.sqrt(cx**2+cy**2) - r



    return d

def cal_pt(r, charge):
    pt = (r*100) / 333.564


    return pt

def display_data(data):
    fig = plt.figure(figsize=(20, 8))
    axs1 = fig.add_subplot(121)
    axs1 = sns.histplot(data['d'], stat='proportion')
    axs1.set(xlabel = 'distance_pre (m)', xlim=(-2.5,0.4))
    axs2 = fig.add_subplot(122)
    axs2 = sns.histplot(data['d_truth'], stat='proportion')
    axs2.set(xlabel='distance_truth (m)', xlim=(-0.05, 0.05))
    #axs3 =

    plt.savefig('distance_v2.png')
    plt.close(fig)

    fig = plt.figure(figsize=(30, 8))
    axs1 = fig.add_subplot(131)
    axs1 = sns.distplot(data['pt'])
    axs1.set(xlabel='pt_pre (GeV)')
    axs2 = fig.add_subplot(132)
    axs2 = sns.distplot(data['pt_truth'])
    axs2.set(xlabel='pt_truth (GeV)')
    axs3 = fig.add_subplot(133)
    axs3 = sns.distplot(data['pt'] - data['pt_truth'])
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

    plt.savefig('pt_v5.png')
    plt.close(fig)



def process(hit, track_gnd, track_pre, input_dir):
    hit_grouped = hit.groupby('event')
    track_gnd_grouped = track_gnd.groupby('event')
    track_pre_grouped = track_pre.groupby('event')

    data = pd.DataFrame(columns=['event', 'd','d_truth', 'pt', 'pt_truth'])
    eff_data = pd.DataFrame(columns = ['event', 'trackId_gnd', 'match_pred', 'total_hits', 'match_hits', 'eff'])
    for event_id, event in hit_grouped:
        #print(event_id)
        event_track_gnd = track_gnd_grouped.get_group(event_id)
        event_track_pre = track_pre_grouped.get_group(event_id)
        event_track_gnd = event_track_gnd.reset_index(drop=True)
        event_track_pre = event_track_pre.reset_index(drop=True)
        event_track_gnd = event_track_gnd.reset_index()
        event_track_pre = event_track_pre.reset_index()

        event_track_gnd = event_track_gnd.rename(columns={'index':'trackId_gnd'})
        event_track_pre = event_track_pre.rename(columns={'index':'match_pred'})

        gnd_base, pre_base , gnd_base_2= match_track(event)

        #gnd_base = gnd_base.drop(gnd_base[gnd_base.match_pred == -1].index)
        #match_pre = gnd_base['match_pred'].values.astype(int).tolist()

        match_pre = event_track_pre.merge(gnd_base, on='match_pred')
        #event_track_pre_left = event_track_pre.iloc[match_pre, :]
        keys = ['cx', 'cy', 'r', 'charge','trackId_gnd']
        track_merge = event_track_gnd[keys].merge(match_pre[keys].reset_index(), on='trackId_gnd', suffixes=('_gnd', '_pre'))

        eff = gnd_base_2.merge(event_track_gnd, on='trackId_gnd')
        if (eff.shape[0]==0):
            print(event_id, 'no track')
            continue

        pt_truth = eff.apply(lambda row: cal_pt(row['r'], row['charge']), axis=1)
        eff['pt_truth'] = pt_truth

        track_merge = track_merge.drop(track_merge[track_merge.trackId_gnd == 0 ].index)
        #track_merge = track_merge.drop(track_merge[track_merge.r_pre < 0.06].index)
        #track_merge = track_merge.drop(track_merge[track_merge.r_gnd < 0.06].index)

        if (track_merge.shape[0]==0):
            print(event_id, 'no track')
            continue

        d = track_merge.apply(lambda row: cal_d(row['cx_pre'], row['cy_pre'], row['r_pre']), axis=1)
        d_truth = track_merge.apply(lambda row: cal_d(row['cx_gnd'], row['cy_gnd'], row['r_gnd']), axis=1)
        pt = track_merge.apply(lambda row: cal_pt(row['r_pre'], row['charge_pre']), axis=1)
        pt_truth = track_merge.apply(lambda row: cal_pt(row['r_gnd'], row['charge_gnd']), axis=1)

        track_merge['d'] = d
        track_merge['d_truth'] = d_truth
        track_merge['pt'] = pt
        track_merge['pt_truth'] = pt_truth
        track_merge['event'] = event_id

        event_data = track_merge[['event', 'd','d_truth', 'pt', 'pt_truth']]

        data = pd.concat([data, event_data])



        eff_data = pd.concat([eff_data, eff])

        #if (event_id>200):
        #    break

    data.to_csv(input_dir+'/match_track_data.csv')
    eff_data.to_csv(input_dir+'/track_eff_data.csv')
    #display_data(data)
    #print('test')

def main():

    input_dir = './results/h2t_polar_distance_v2/prediction'
    hit_file = input_dir + '/'+'hits_prediction_test.csv'
    track_gnd_file = input_dir + '/'+'param_gnd_test.csv'
    track_pred_file = input_dir + '/'+'param_prediction_test.csv'

    hit = pd.read_csv(hit_file, index_col=0)
    track_gnd = pd.read_csv(track_gnd_file, index_col=0)
    track_pre = pd.read_csv(track_pred_file, index_col=0)

    #data = pd.read_csv(input_dir+'/match_track_data.csv')
    #display_data(data)
    process(hit, track_gnd, track_pre, input_dir)

 # wirete






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


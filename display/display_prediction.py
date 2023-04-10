import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle

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
    eff_sample = pd.DataFrame(columns=['trackId_gnd', 'match_pred', 'total_hits', 'match_hits', 'eff', 'match_pred_2'])
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
            eff_match_pred_2 = -1

        else:
            match_track = match_grouped.get_group(track_id)
            for id, tk in match_track.iterrows():
                sum +=val_count[tk['pred']]
                eff_match_pred = eff_match_pred + str(tk['pred']) +','
                eff_match_pred_2 = tk['pred']
            eff = sum/val_count.sum()
        track_eff_sum += eff
        count_gnd += 1
        eff_df.loc[len(eff_df.index)] = [track_id, round(eff,3)]
        eff_sample.loc[len(eff_sample.index)] = [int(track_id), eff_match_pred, val_count.sum(), sum, round(eff,3), eff_match_pred_2]


    track_eff = track_eff_sum / count_gnd

    match.sort_values(by='gnd')
    return track_eff, track_purity, eff_df, purity_df, eff_sample, purity_sample, hits, match


def draw_result(result,  pdf, n,circle_preds):

    draw_event = n

    pred_grouped = result.groupby('event')
    circle_grouped = circle_preds.groupby('event')

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

        #不把噪声那一类放进track_eff统计
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
        event_circle = circle_grouped.get_group(event_id)
        event_circle = event_circle.reset_index()
        match_preds = eff_sample['match_pred_2']
        match_preds = match_preds.drop(match_preds[match_preds<0].index)

        event_circle = event_circle.loc[match_preds.values, :]
        left_cols = ['cx', 'cy', 'r']
        event_circle_2 = event_circle[left_cols]
        match_hit += eff_sample['match_hits'].sum()
        sum_hit +=eff_sample['total_hits'].sum()
        # ((eff_sample.shape[0] + 1 != purity_sample.shape[0]) & (count < 200))
        if(count == 7):
            print('dubug')
        if (count<draw_event ):
            if (count==71):
                print("71 event")
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
            #axs2.table(cellText=purity_df.values, colLabels=purity_df.columns, colWidths=[0.1,0.1], loc='lower left')

            eff_colwitdth = [0.18] *eff_sample.shape[1]
            purity_colwitdth = [0.18] * purity_sample.shape[1]

            axs2.table(cellText=eff_sample.values, colLabels=eff_sample.columns,colWidths=eff_colwitdth, loc='upper left')
            if (len(purity_sample>0)):
                axs2.table(cellText=purity_sample.values, colLabels=purity_sample.columns, colWidths=purity_colwitdth, loc='lower left')
            axs1.table(cellText=match.values, colLabels=match.columns, colWidths=[0.1, 0.1], loc='upper center')

            #for cindex, circle in event_circle_2.iterrows():
            #    circle = Circle([circle['cx'], circle['cy']], circle['r'], fill=False, alpha=0.8)
           #     axs2.add_patch(circle)

            #if (len(eff_df.index)!=len(purity_df.index)):
            #    axs2.text(-0.4, -0.7, 'Quantity of prediction trackIndex is wrong!', c='red')
            plt.title('event ' + str(int(event_id+1)) + ' Prediction')
            pdf.savefig()
            plt.close()
        if (len(eff_df.index) != len(purity_df.index)):
            wrong_count += 1

        #if(count>100):
        #    break
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


def main():

    input_dir = './results/h2t_polar_distance_v1/prediction'
    file_name = '/hits_prediction_test_rc'

    circle_preds_file = input_dir + '\param_prediction_test.csv'

    input_file = input_dir + file_name + '.csv'
    output = input_dir+ '/hits_prediction_display.pdf'


    result = pd.read_csv(input_file)
    circle_preds = pd.read_csv(circle_preds_file)
    draw_event = 50

    #val_count = result.value_counts('trackId_gnd')
    #ratio = val_count[0] / val_count.sum()
    #print('noise:' ratio)


    with PdfPages(output) as pdf:
        draw_result(result, pdf, draw_event,circle_preds)
        print('Save to pdf file',pdf)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


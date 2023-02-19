import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle


def draw_result(hits,param_pred, param_gnd, pdf, draw_event):
    hits_grouped = hits.groupby('event')
    param_pred_grouped = param_pred.groupby('event')
    param_gnd_grouped = param_gnd.groupby('event')

    count = 0
    for event_id, event in hits_grouped:
        print('drawing event ', event_id)


        event_param_pred = param_pred_grouped.get_group(event_id)
        event_param_gnd = param_gnd_grouped.get_group(event_id)

        event_param_gnd = event_param_gnd.reset_index(drop=True)
        event_param_pred = event_param_pred.reset_index(drop=True)

        pred_track = event['trackId_pred'].value_counts()
        gnd_track = event['trackId_gnd'].value_counts()
        pred_track_index = pred_track.index
        gnd_track_index = gnd_track.index

        event['trackId_pred'] = event['trackId_pred'].apply(lambda x: 10 - x)
        fig = plt.figure(figsize=(30, 8))
        axs0 = fig.add_subplot(131)
        outterWall = plt.Circle((0, 0), 0.81, fill=False, alpha=0.2)
        innerWall = plt.Circle((0, 0), 0.06, fill=False, alpha=0.2)
        axs0.add_artist(outterWall)
        axs0.add_artist(innerWall)
        axs0.set_xlim((-0.82, 0.82))
        axs0.set_ylim((-0.82, 0.82))
        axs0.set_aspect(1)
        sc0 = axs0.scatter(event['x'], event['y'], marker='o', alpha= 0.8, c='skyblue')
        plt.title('event ' + str(int(event_id)) + ' Raw Data')

        axs1 = fig.add_subplot(132)
        outterWall = plt.Circle((0, 0), 0.81, fill=False, alpha=0.2)
        innerWall = plt.Circle((0, 0), 0.06, fill=False, alpha=0.2)
        axs1.add_artist(outterWall)
        axs1.add_artist(innerWall)
        axs1.set_xlim((-0.82, 0.82))
        axs1.set_ylim((-0.82, 0.82))
        axs1.set_aspect(1)

        sc1 = axs1.scatter(event['x'], event['y'], marker='o',c=event['trackId_gnd'], cmap=plt.cm.Set1_r, alpha=0.7)
        plt.colorbar(sc1, fraction=0.05)

        for trackId in range(len(event_param_gnd)):
            track = event_param_gnd.loc[trackId,:]
            circle = Circle([track['cx'], track['cy']], track['r'],color='skyblue', fill=False, alpha=0.9, linewidth=3)
            axs1.add_patch(circle)

        plt.title('event ' + str(int(event_id)) + ' Ground Truth')

        axs2 = fig.add_subplot(133)
        outterWall = plt.Circle((0, 0), 0.81, fill=False, alpha=0.2)
        innerWall = plt.Circle((0, 0), 0.06, fill=False, alpha=0.2)
        axs2.add_artist(outterWall)
        axs2.add_artist(innerWall)
        axs2.set_xlim((-0.82, 0.82))
        axs2.set_ylim((-0.82, 0.82))
        axs2.set_aspect(1)

        sc2 = axs2.scatter(event['x'], event['y'], marker='o', c=event['trackId_pred'], cmap=plt.cm.Set1_r, alpha=0.5)
        plt.colorbar(sc2, fraction=0.05)
        for trackId in pred_track_index:
            if trackId == 9:
                continue
            track = event_param_pred.loc[trackId, :]
            circle = Circle([track['cx'], track['cy']], track['r'],color='b',  fill=False, alpha=0.7, linewidth=3)
            axs2.add_patch(circle)


        plt.title('event ' + str(int(event_id)) + ' Prediction')
        pdf.savefig()
        plt.close()

        count += 1
        if(count>draw_event):
            break


def main():

    input_dir = 'E:\ihep\BESIII\hep_track\output_2\h2t_polar\h2t_polar_v3'
    hit_file = input_dir + '\hits_prediction_test_rc.csv'  #_rc = polar to rc
    param_pred_file = input_dir + '\param_prediction_test.csv'
    param_gnd_file = input_dir + '\param_gnd_test.csv'

    hits = pd.read_csv(hit_file, nrows=10*10000)
    param_pred = pd.read_csv(param_pred_file)
    param_gnd = pd.read_csv(param_gnd_file)


    output = 'hits_prediction_polar_v3_with_circles.pdf'


    draw_event = 200

    #val_count = result.value_counts('trackId_gnd')
    #ratio = val_count[0] / val_count.sum()
    #print('noise:' ratio)


    with PdfPages(output) as pdf:
        draw_result(hits,param_pred, param_gnd, pdf, draw_event)
        print('Save to pdf file',pdf)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


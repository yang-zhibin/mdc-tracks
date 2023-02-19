import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
from collections import OrderedDict

def find_circle(track):
    flight_len = 1e5
    for index, hit in track.iterrows():
        if flight_len > hit['flightLength']:
            flight_len = hit['flightLength']
            circle_index = index


    return circle_index


    print()

def drawEvents(data, drop_radius, pdf,n):
    data_grouped  = data.groupby('event')
    index = 1
    for event_id, event in data_grouped:
        print('drawing event', event_id)
        event['trackIndex'] = event['trackIndex'].apply(lambda x: x-5 if x > 5 else x)
        fig = plt.figure(figsize=(20, 8))

        axs1 = fig.add_subplot(131)


        outterWall = plt.Circle( (0, 0 ),81,fill = False, alpha=0.4 )
        innerWall = plt.Circle( (0, 0 ),6,fill = False, alpha=0.4 )
        axs1.add_artist( outterWall )
        axs1.add_artist( innerWall )
        axs1.set_xlim((-82, 82))
        axs1.set_ylim((-82, 82))
        axs1.set_aspect(1)#xy ratio 1:1

        event['currentTrackPID'] = event['currentTrackPID'].apply(lambda x: x+9500 if x==-9999 else x )
        track_data = event.groupby('trackIndex')


        sc1 = axs1.scatter(event['x'], event['y'], marker='o', s=abs(event['rawDriftTime']) / 25, alpha= 0.8, c='skyblue')
        #plt.colorbar(sc1, fraction=0.05)

        plt.title('event ' + str(event_id) + ' raw hits')



        axs2 = fig.add_subplot(133)
        outterWall = plt.Circle((0, 0), 81, fill=False, alpha=0.4)
        innerWall = plt.Circle((0, 0), 6, fill=False, alpha=0.4)
        axs2.add_artist(outterWall)
        axs2.add_artist(innerWall)
        axs2.set_xlim((-82, 82))
        axs2.set_ylim((-82, 82))
        axs2.set_aspect(1)
        sc2 = axs2.scatter(event['x'], event['y'], marker='o' , c=event['trackIndex'],  cmap=plt.cm.Dark2_r, alpha=0.8, label=event['trackIndex'])
        plt.colorbar(sc2, fraction=0.05)

        for trackId, track in track_data:
            charge = track['chargeParticle'].values[0]
            if trackId >= 0:

                particle_value = track['currentTrackPID'].values[0]

                circle_index = track['flightLength'].idxmin()
                cx = track.loc[circle_index, 'centerX']
                cy = track.loc[circle_index, 'centerY']
                r = track.loc[circle_index, 'radius']
                x = track.loc[circle_index, 'x']
                y = track.loc[circle_index, 'y']

                #axs2.text(x, y, s='pID: ' + str(particle_value), fontsize=5, alpha=0.7)

                if charge < 0:
                    circleColor = 'pink'
                    #axs2.scatter(x, y, marker='o', s=5, c=circleColor, alpha=0.7)

                    circle = Circle([cx, cy], r, color=circleColor, fill=False, alpha=0.8, linewidth=3)
                    axs2.add_patch(circle)

                else:
                    circleColor = 'skyblue'
                   # axs2.scatter(x, y, marker='o', s=5, c=circleColor, alpha=0.7)

                    # 带正电粒子径迹圆原始数据
                    #circle = Circle([cx, cy], r, color=circleColor, fill=False, alpha=0.5)
                    #axs2.add_patch(circle)

                    #修正后的数据，修正方法：点对称
                    circle = Circle([x - (cx - x), y - (cy - y)], r, color=circleColor, fill=False, alpha=0.8, linewidth=3)
                    axs2.add_patch(circle)

        plt.title('event ' + str(event_id) + ' TrackIndex with track circle')

        axs3 = fig.add_subplot(132)
        outterWall = plt.Circle((0, 0), 81, fill=False, alpha=0.4)
        innerWall = plt.Circle((0, 0), 6, fill=False, alpha=0.4)
        axs3.add_artist(outterWall)
        axs3.add_artist(innerWall)
        axs3.set_xlim((-82, 82))
        axs3.set_ylim((-82, 82))
        axs3.set_aspect(1)
        sc3 = axs3.scatter(event['x'], event['y'], marker='o', c=event['trackIndex'], cmap=plt.cm.Dark2_r, alpha=0.8,
                           label=event['trackIndex'])
        plt.colorbar(sc3, fraction=0.05)

        plt.title('event ' + str(event_id) + ' TrackIndex ')
        '''
        axs3 = fig.add_subplot(133)
        #outterWall = plt.Circle((0, 0), 81, fill=False, alpha=0.2)
        innerWall = plt.Circle((0, 0), 6, fill=False, alpha=0.2)
        #axs3.add_artist(outterWall)
        axs3.add_artist(innerWall)
        axs3.set_xlim((-82, 82))
        axs3.set_ylim((-82, 82))
        axs3.set_aspect(1)

        for trackId, track in track_data:
            track = track.drop(track[track.radius < drop_radius].index)
            if (track.shape[0] < 1 and trackId > 0):
                print('Abnormal in event id:', event_id, 'track id:', trackId, 'drop radius:', drop_radius)
                continue
            elif(track.shape[0] < 1):
                continue
            charge = track['chargeParticle'].values[0]
            if trackId >= 0:

                particle_value = track['currentTrackPID'].values[0]

                circle_index = track['flightLength'].idxmin()
                cx = track.loc[circle_index, 'centerX']
                cy = track.loc[circle_index, 'centerY']
                r = track.loc[circle_index, 'radius']
                x = track.loc[circle_index, 'x']
                y = track.loc[circle_index, 'y']

                #axs3.text(x, y, s='pID: ' + str(particle_value), fontsize=5, alpha=1)

                if charge < 0:
                    circleColor = 'pink'
                    axs3.scatter(x, y, marker='o', s=2, c=circleColor, alpha=1)

                    circle = Circle([cx, cy], r, color=circleColor, fill=False, alpha=1)
                    axs3.add_patch(circle)

                else:
                    circleColor = 'skyblue'
                    axs3.scatter(x, y, marker='o', s=2, c=circleColor, alpha=1)

                    # 带正电粒子径迹圆原始数据
                    # circle = Circle([cx, cy], r, color=circleColor, fill=False, alpha=0.5)
                    # axs1.add_patch(circle)

                    # 修正后的数据，修正方法：点对称
                    circle = Circle([x - (cx - x), y - (cy - y)], r, color=circleColor, fill=False, alpha=1)
                    axs3.add_patch(circle)
        #handles, labels = plt.gca().get_legend_handles_labels()
        #by_label = OrderedDict(zip(labels, handles))
        #plt.legend(by_label.values(), by_label.keys())
        sc3 = axs3.scatter(event['x'], event['y'], marker='o',
                           c=event['trackIndex'], cmap='rainbow', alpha=0.5, label=event['currentTrackPID'])
        plt.colorbar(sc3, fraction=0.05)
        plt.title('event ' + str(event_id) + ' Circle Radius > 6')
        '''


        pdf.savefig()
        if n and index >= n - 1:
            break
        plt.close()
        index += 1

def main():
    input_file = 'E:\graph_ML\BESIII\data\data_processed\mdcDigi_1_0_cleaned_5.csv'
    #output_file = 'display_circle.pdf'

    data = pd.read_csv(input_file, nrows=50*10000,)

    output = 'display_circle_output/display_gnd_pre_v5.pdf'

    print("Input file:", input_file)
    print("Output file:", output)
    drop_radius = 6
    n = 20
    with PdfPages(output) as pdf:
        drawEvents(data, drop_radius, pdf,n)
        print('Save to pdf file',pdf)




if __name__ == '__main__':
    main()

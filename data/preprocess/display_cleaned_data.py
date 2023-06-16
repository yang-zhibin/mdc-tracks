import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def drawEvents(data, raw_data, pdf, n):
    data_grouped = data.groupby('event')
    raw_data_grouped = raw_data.groupby('event')
    index = 1
    for event_id, event in data_grouped:
        raw_event = raw_data_grouped.get_group(event_id)
        event = event.drop(event[event.currentTrackPID ==-9999].index)
        raw_event = raw_event.drop(raw_event[raw_event.currentTrackPID == -9999].index)
        print('drawing event', event_id)
        fig = plt.figure(figsize=(20, 8))

        axs1 = fig.add_subplot(131)
        outterWall = plt.Circle((0, 0), 81, fill=False, alpha=0.2)
        innerWall = plt.Circle((0, 0), 6, fill=False, alpha=0.2)
        axs1.add_artist(outterWall)
        axs1.add_artist(innerWall)
        axs1.set_xlim((-82, 82))
        axs1.set_ylim((-82, 82))
        axs1.set_aspect(1)
        sc1 = axs1.scatter(event['x'], event['y'], marker='o',
                           c=event['trackIndex'], cmap=plt.cm.tab20b, alpha=0.7, label=event['trackIndex'])
        plt.colorbar(sc1, fraction=0.05)


        plt.title('event ' + str(event_id) + ' cleaned trackId')

        axs2 = fig.add_subplot(133)

        outterWall = plt.Circle((0, 0), 81, fill=False, alpha=0.2)
        innerWall = plt.Circle((0, 0), 6, fill=False, alpha=0.2)
        axs2.add_artist(outterWall)
        axs2.add_artist(innerWall)
        axs2.set_xlim((-82, 82))
        axs2.set_ylim((-82, 82))
        axs2.set_aspect(1)  # xy ratio 1:1

        sc2 = axs2.scatter(raw_event['x'], raw_event['y'], marker='o', s=abs(raw_event['rawDriftTime']) / 25,
                           c=raw_event['currentTrackPID'], cmap='rainbow', alpha=0.7)
        plt.colorbar(sc2, fraction=0.05)

        plt.title('event ' + str(event_id) + ' Particle Id')

        axs3 = fig.add_subplot(132)

        outterWall = plt.Circle((0, 0), 81, fill=False, alpha=0.2)
        innerWall = plt.Circle((0, 0), 6, fill=False, alpha=0.2)
        axs3.add_artist(outterWall)
        axs3.add_artist(innerWall)
        axs3.set_xlim((-82, 82))
        axs3.set_ylim((-82, 82))
        axs3.set_aspect(1)  # xy ratio 1:1

        sc3 = axs3.scatter(raw_event['x'], raw_event['y'], marker='o',
                           c=raw_event['trackIndex'], cmap=plt.cm.tab20b, alpha=0.7)
        plt.colorbar(sc3, fraction=0.05)
        plt.title('event ' + str(event_id) + ' raw trackId')


        pdf.savefig()
        if n and index >= n - 1:
            break
        plt.close()
        index += 1
        
def drawCleanEvents(data, pdf, n):
    data_grouped = data.groupby('event')
    
    index = 1
    for event_id, event in data_grouped:
        
        #event = event.drop(event[event.currentTrackPID ==-9999].index)
        
        print('drawing event', event_id)
        fig = plt.figure(figsize=(20, 8))

        axs1 = fig.add_subplot(111)
        outterWall = plt.Circle((0, 0), 81, fill=False, alpha=0.2)
        innerWall = plt.Circle((0, 0), 6, fill=False, alpha=0.2)
        axs1.add_artist(outterWall)
        axs1.add_artist(innerWall)
        axs1.set_xlim((-82, 82))
        axs1.set_ylim((-82, 82))
        axs1.set_aspect(1)
        sc1 = axs1.scatter(event['x'], event['y'], marker='o', c=event['trackIndex'], cmap=plt.cm.tab20b, alpha=0.7, label=event['trackIndex'])
        plt.colorbar(sc1, fraction=0.05)


        plt.title('event ' + str(event_id) + ' cleaned trackId')



        pdf.savefig()
        if n and index >= n - 1:
            break
        plt.close()
        index += 1

if __name__ == '__main__':

    input_file = "./data/rawData/pipijpsi/pipijpsi_mdcDigiMc_0.csv"

    output = './data/processedData/rhoPi/display_pipijpsi_raw_data.pdf'

    #raw_data_file = "D:\ihep\mdc-tracks\data\rawData\rhoPi\mdcDigiMc_1_0.csv"
    
    #wirePos = "D:\ihep\mdc-tracks\data\preprocess\MdcWirePosition.csv"

    print('input file name:', input_file)
    print("output file name:",output)
    rows = 50 * 10000
    data = pd.read_csv(input_file)
    #raw_data = pd.read_csv(raw_data_file, nrows=rows)
    
    
    


    n=500

    with PdfPages(output) as pdf:
        drawCleanEvents(data,  pdf, n)
        print('Save to pdf file', pdf)
        
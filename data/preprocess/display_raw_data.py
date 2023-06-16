import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math
from matplotlib.patches import Circle
from pylab import *

def cal_phi(x, y):
    phi = math.atan2(y, x)
    return phi

def cal_r(x, y):
    r = math.sqrt(x*x+y*y)
    return r

def load_wirePos(wirePos_file):
    print("wire position file: " + wirePos_file)
    wirePos = pd.read_csv(wirePos_file)
    wirePos['x'] = (wirePos['east_x'] + wirePos['west_x'])/2.
    wirePos['y'] = (wirePos['east_y'] + wirePos['west_y']) / 2.

    phi = wirePos.apply(lambda row: cal_phi(row['x'], row['y']), axis=1)
    r = wirePos.apply(lambda row: cal_r(row['x'], row['y']), axis=1)

    wirePos['phi'] = phi
    wirePos['r'] = r

    wirePos = wirePos.drop(columns=['layer', 'cell'])
    return wirePos

def charge_from_PID(pid):
    if pid == 11:
        return -1
    elif pid == -11:
        return 1
    elif pid == 13:
        return -1   
    elif pid == -13:    
        return 1
    elif pid < 0:
        return -1
    elif pid > 0:
        return 1


def load_mcParticle(mcParticle_file):
    print("mcParticle file: " + mcParticle_file)
    mcParticle = pd.read_csv(mcParticle_file, index_col=False)
    mcParticle.loc[:, 'pt'] = mcParticle.apply(lambda row: cal_r(row['initialFourMomentumx'], row['initialFourMomentumy']), axis=1)
    mcParticle.loc[:, 'phi'] = mcParticle.apply(lambda row: math.atan2(row['initialFourMomentumy'], row['initialFourMomentumx']), axis=1)

    mcParticle.loc[:, 'radius'] = mcParticle.apply(lambda row: row['pt']*333.564, axis=1)

    
    mcParticle.loc[:, 'charge'] = mcParticle['particleProperty'].apply(lambda x: charge_from_PID(x))
    
    mcParticle.loc[:, 'angle'] = mcParticle.apply(lambda row: math.atan2(row['initialFourMomentumx'], -row['initialFourMomentumy']), axis=1)
    
    mcParticle.loc[:, 'cx'] = mcParticle.apply(lambda row: row['initialPositionx'] + row['charge']*row['radius']*math.cos(row['angle']), axis=1)
    mcParticle.loc[:, 'cy'] = mcParticle.apply(lambda row: row['initialPositiony'] + row['charge']*row['radius']*math.sin(row['angle']), axis=1)
    
    return mcParticle
def allocate_wirePos(wirePos, rawData):

    # fix track index bug from raw data
    rawData['trackIndex'] = rawData['trackIndex'].apply(lambda x: x-1000 if x >= 1000 else x)
    rawData['trackIndex'] = rawData['trackIndex'].apply(lambda x: -1 if x < 0 else x)
    
    
    rawData['currentTrackPID'] = rawData['currentTrackPID'].apply(lambda x: 0 if x==-9999 else x)

    #allocate wire position through gid
    hits = pd.merge(rawData, wirePos, on='gid')

    #calculate pt
    pt = hits.apply(lambda row: cal_r(row['momx'], row['momy']), axis=1)
    hits.loc[:, 'pt'] = pt

    #hits = hits.sort_values(by='event')

    left_cols = ['event', 'gid', 'layer', 'cell', 'x', 'y', 'r', 'phi', 'trackIndex', 'centerX', 'centerY',	'radius','momx','momy','momz', 'chargeParticle', 'flightLength', 'currentTrackPID', 'rawDriftTime', 'creatorProcessturnID', 'isScondary']

    hits = hits[left_cols]

    return hits

def drawRawEvents(data, wirePos, mcParticle, pdf, n):
    data_grouped = data.groupby('event')
    mcParticle_grouped = mcParticle.groupby('event')
    index = 1
    for event_id, event in data_grouped:
        event = allocate_wirePos(wirePos, event)
        Particle = mcParticle_grouped.get_group(event_id)
        event = event.drop(event[event.trackIndex == -1].index)
        #Particle = Particle.groupby('trackIndex')
        #event = event.drop(event[event.currentTrackPID ==-9999].index)
        
        print('drawing event', event_id)
        fig = plt.figure(figsize=(10, 8))

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
        trackIdCount = event.value_counts('trackIndex')
        
        for trackid, track in Particle.iterrows():
            #if(track['trackIndex'] in [7,8,9, 10]):
            if(track['trackIndex'] in trackIdCount.index.tolist()):
                if (track['charge']>0):
                    trackColor = 'skyblue'
                    hatch = '/'
                else:
                    trackColor = 'pink'
                    hatch = '\\'
                circle = Circle((track['cx'], track['cy']), track['radius'], color = trackColor, fill=False, alpha=0.8, linewidth=3, hatch=hatch)
                axs1.add_patch(circle)
            
            
        plt.title('event ' + str(event_id) + ' raw trackId')



        pdf.savefig()
        if n and index >= n - 1:
            break
        plt.close()
        index += 1

if __name__ == '__main__':

    input_file = "./data/rawData/rhoPi/rhoPi_mdcDigiMc_0.csv"

    output = './data/rawData/rhoPi/display_rhoPi_raw_data.pdf'

    mcParticle_file = './data/rawData\\rhoPi\\rhoPi_mcParticle_0.csv'
    
    wirePos_file = "D:\ihep\mdc-tracks\data\preprocess\MdcWirePosition.csv"
    wirePos = load_wirePos(wirePos_file)

    print('input file name:', input_file)
    print("output file name:",output)
    rows = 10 * 10000
    data = pd.read_csv(input_file, index_col=False, nrows=rows)
    
    mcParticle = load_mcParticle(mcParticle_file)
    #raw_data = pd.read_csv(raw_data_file, nrows=rows)
    
    n=100

    with PdfPages(output) as pdf:
        drawRawEvents(data, wirePos, mcParticle, pdf, n)
        print('Save to pdf file', pdf)
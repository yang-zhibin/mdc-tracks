import math

import pandas as pd
import numpy as np
import time

def process_chunk(chunk, wirePos):
    chunk['trackIndex'] = chunk['trackIndex'].apply(lambda x: x-1000 if x >= 1000 else x)
    chunk['trackIndex'] = chunk['trackIndex'].apply(lambda x: -1 if x < 0 else x)
    #chunk['currentTrackPID']

    hits = pd.merge(chunk, wirePos, on='gid')
    print('chunk shape', hits.shape)

    return hits

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

    wirePos = wirePos.rename(columns={'id':'gid'})
    return wirePos




def process_mdcDigi(mdcDigi_file, wirePos_file, output_file, left_cols):
    print("processing mdcDigi: " + mdcDigi_file)


    wirePos = load_wirePos(wirePos_file)

    chunksize = 50*10000

    time0 = time.time()
    digi = pd.read_csv(mdcDigi_file, header=0, chunksize=chunksize, index_col=False)
    index = 1
    for chunk in digi:
        print("processing chunk", index, "max event:", chunk['event'].max())
        t0 = time.time()
        digi_hits = process_chunk(chunk, wirePos)
        digi_hits = digi_hits.rename(columns={'layer_x':'layer', 'cell_x':'cell'})
        hits = digi_hits[left_cols]
        #phi = hits.apply(lambda row: cal_phi(row['x'], row['y']), axis=1)
        #r = hits.apply(lambda row: cal_r(row['x'], row['y']), axis=1)
        pt = hits.apply(lambda row: cal_r(row['momx'], row['momy']), axis=1)

        #hits['phi'] = phi
        #hits['r'] = r
        hits['pt'] = pt

        hits = hits.sort_values(by='event')
        if (index == 1):
            hits.to_csv(output_file, index=False)
            print('output shape', hits.shape)
        else:
            hits.to_csv(output_file, mode='a', header=False, index=False)
            print('output shape', hits.shape)

        index += 1
        t1 = time.time()
        print('This chunk data processed in', t1 - t0, 'seconds')
    time1 = time.time()
    print('All data processed in', time1 - time0, 'seconds')

if __name__ == '__main__':
    mdcDigi_file = "E:\graph_ML\BESIII\data\mdcDigiMc_1_0.csv"
    wirePos_file = "MdcWirePosition.csv"
    output_dir = "E:\graph_ML\BESIII\data\data_processed"

    output_file = output_dir + "\mdcDigi_1_0_circle_5.csv"

    left_cols = ['event', 'gid', 'layer', 'cell', 'x', 'y', 'r', 'phi', 'trackIndex', 'centerX', 'centerY',	'radius','momx','momy','momz', 'chargeParticle', 'flightLength', 'currentTrackPID', 'rawDriftTime', 'creatorProcessturnID']


    print("output file name:",output_file)
    print("left columns", left_cols)

    process_mdcDigi(mdcDigi_file, wirePos_file, output_file, left_cols)

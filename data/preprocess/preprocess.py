import os
import argparse
import pandas as pd
import math

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

def allocate_wirePos(wirePos, rawData):

    # fix track index bug from raw data
    rawData['trackIndex'] = rawData['trackIndex'].apply(lambda x: x-1000 if x >= 1000 else x)
    rawData['trackIndex'] = rawData['trackIndex'].apply(lambda x: -1 if x < 0 else x)

    #allocate wire position through gid
    hits = pd.merge(rawData, wirePos, on='gid')

    #calculate pt
    pt = hits.apply(lambda row: cal_r(row['momx'], row['momy']), axis=1)
    hits.loc[:, 'pt'] = pt

    hits = hits.sort_values(by='event')

    left_cols = ['event', 'gid', 'layer', 'cell', 'x', 'y', 'r', 'phi', 'trackIndex', 'centerX', 'centerY',	'radius','momx','momy','momz', 'chargeParticle', 'flightLength', 'currentTrackPID', 'rawDriftTime', 'creatorProcessturnID', 'isScondary']

    hits = hits[left_cols]

    return hits

def clean_data(hits, args)

def process_file(data_file, out_dir, wirePos_file, args):
    # load wire position and calculate (phi, r) from (x, y)
    wirePos = load_wirePos(wirePos_file)
    rawData = pd.read_csv(data_file)


    hits = allocate_wirePos(wirePos, rawData) 
    hits = clean_data(hits, args)


def main():
    parser = argparse.ArgumentParser(description = 'MDC data preprocess implementation')
    parser.add_argument('--rawDriftTime-max', type=float, default=1600)

    args = parser.parse_args()

    rawData_dir = "./data/rawData/rhoPi"
    out_dir = "./data/processedData/rhoPi"
    wirePos_file = "MdcWirePosition.csv"

    for fname in os.listdir(rawData_dir):
        if fname.endswith(".csv"):
            print("Processing file: " + fname)
            process_file (os.path.join(rawData_dir, fname), out_dir, wirePos_file, args)

if __name__ == '__main__':
    main()
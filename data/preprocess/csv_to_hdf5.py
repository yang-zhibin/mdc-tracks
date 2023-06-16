import pandas as pd
import h5py
import os
import math
import numpy as np

def cal_phi(x, y):
    phi = math.atan2(y, x)
    return phi

def cal_r(x, y):
    r = math.sqrt(x*x+y*y)
    return r

def load_wirePos(wirePos_file):
    print("wire position file: " + wirePos_file)
    wirePos = pd.read_csv(wirePos_file, index_col=False)
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

    #left_cols = ['event', 'gid', 'layer', 'cell', 'x', 'y', 'r', 'phi', 'trackIndex', 'centerX', 'centerY',	'radius','momx','momy','momz', 'chargeParticle', 'flightLength', 'currentTrackPID', 'rawDriftTime', 'creatorProcessturnID', 'isScondary']

    #hits = hits[left_cols]

    return hits

def process(out_dir, wirePos, mcPariticle, mdcDigiMc):
    print('processing...')
    mcPariticle_run = mcPariticle.groupby('run')
    mdcDigiMc_run = mdcDigiMc.groupby('run')
    
    eventType = 'pipijpsi'
    eventCount = 0
    fileSize = 1000
    fileNo = 3
    
    hit_feature_col = ['x', 'y', 'r', 'phi', 'rawDriftTime']
    hit_label_col = ['trackIndex', 'driftDistance']
    track_label_col = ['trackIndex', 'cx', 'cy', 'radius', 'charge']
    
    event_info_col = [ 'event_type', 'run', 'event']
    hit_info_col = ['gid']
    track_info_col = ['particleProperty']

    out_file = out_dir + '/' + 'bes3data' + '_' + '%04d' % fileNo + '.hdf5'
    print('out_file: ' + out_file)
    f = h5py.File(out_file, 'w')
    #group_paeticle = f.create_group(eventType)
    
    for runId, run in mdcDigiMc_run:
        print('runId: ' + str(runId))
        mdcDigiMc_event = run.groupby('event')
        mcPariticle_event = mcPariticle_run.get_group(runId).groupby('event')
        
        #group_run = group_paeticle.create_group('run'+str(runId))
        
        #group_run.create_dataset('hit_feature', shape=[0,0,4],max)
                
        for eventId, event in mdcDigiMc_event:
            print('eventId: ' + str(int(eventId)))
            
            group_event =f.create_group('event_'+'%06d'%eventCount)
            # group_event =group_run[str(eventId+1e5)] = eventId
            
            event = allocate_wirePos(wirePos, event)
            mcPariticle_track = mcPariticle_event.get_group(eventId)
            trackIndex = event['trackIndex'].value_counts().index.tolist()
            mcPariticle_track = mcPariticle_track[mcPariticle_track.trackIndex.isin(trackIndex)]
            
            
            hit_feature = event[hit_feature_col]
            hit_label = event[hit_label_col]
            track_label = mcPariticle_track[track_label_col]
            
            hit_info = event[hit_info_col]
            track_info = mcPariticle_track[track_info_col]
            
            event_info_dtype =np.dtype({'names': event_info_col, 'formats': ['S10','i4', 'i4']})
            
            event_info = np.array([(eventType, runId, eventId)], dtype=event_info_dtype)
            
            group_event.create_dataset('hit_feature', data=hit_feature.to_records(index=False))
            group_event.create_dataset('hit_label', data=hit_label.to_records(index=False))
            group_event.create_dataset('track_label', data=track_label.to_records(index=False))
            group_event.create_dataset('hit_info', data=hit_info.to_records(index=False))
            group_event.create_dataset('track_info', data=track_info.to_records(index=False))
            group_event.create_dataset('event_info', data=event_info)
            
            
            eventCount += 1
            if eventCount >100:
               break
            

def main():
    rawData_dir = "./data/rawData/pipijpsi"
    out_dir = "./data/HDF5/pipijpsi"
    wirePos_file = "./data/preprocess/MdcWirePosition.csv"
    
    mcPariticle_file =rawData_dir + "/pipijpsi_mcParticle_0.csv"
    mdcDigiMc_file = rawData_dir + "/pipijpsi_mdcDigiMc_0.csv"
    
    wirePos = load_wirePos(wirePos_file)
    
    mcPariticle = load_mcParticle(mcPariticle_file)
    mdcDigiMc = pd.read_csv(mdcDigiMc_file, index_col=False)
    
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    process(out_dir, wirePos, mcPariticle, mdcDigiMc)
    
    
if __name__ == '__main__':
    main()
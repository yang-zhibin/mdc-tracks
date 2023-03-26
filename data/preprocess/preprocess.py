import os
import argparse
import pandas as pd
import math
import time

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

    #hits = hits.sort_values(by='event')

    left_cols = ['event', 'gid', 'layer', 'cell', 'x', 'y', 'r', 'phi', 'trackIndex', 'centerX', 'centerY',	'radius','momx','momy','momz', 'chargeParticle', 'flightLength', 'currentTrackPID', 'rawDriftTime', 'creatorProcessturnID', 'isScondary']

    hits = hits[left_cols]

    return hits

def clean_data(rawData, wirePos, args, clean_out_file, problem_out_file):
    
    data_grouped = rawData.groupby('event')

    problem_count = pd.Series(
        {'total': 0,'total_hits<10': 0, 'pid_max_part<0.9': 0, 'gap_layer>5': 0, 'min_layer>8': 0, 'cross_layer<8': 0,
         'creator_noise': 0, 'creator_drop': 0, 'center_drop':0})
    #clean_data = pd.DataFrame(columns=list(hits))
    #problem_data = pd.DataFrame(columns=list(hits))
    #problem_data['problem'] = 0
    count = 0
    for event_id, event in data_grouped:
        event = allocate_wirePos(wirePos, event)
        track_grouped = event.groupby('trackIndex')
        noise_problem = False
        drop_problem = False
        event_process = event
        problem = 'Problem description: \n'
        for trackId, track in track_grouped:
            if (trackId == -1):
                continue

            min_layer = track['layer'].min()
            if (min_layer > 8):
                event_process = event_process.drop(event_process[event_process.trackIndex == trackId].index)
                drop_problem = True
                problem = problem + 'trackId: ' + str(trackId) + ' ' + 'min_layer>8 ' + '\n'
                problem_count['min_layer>8'] = problem_count['min_layer>8'] + 1
                continue

            problem_count['total'] = problem_count['total'] + 1
            hit_count = track.shape[0]
            if (hit_count < 10):
                event_process.loc[(event_process[event_process.trackIndex == trackId].index), 'trackIndex'] = -1
                noise_problem = True
                problem = problem + 'trackId: ' + str(trackId) + ' ' + 'total_hits<10 ' + '\n'
                problem_count['total_hits<10'] = problem_count['total_hits<10'] + 1
                continue



            layer_count = track.value_counts('layer')
            layer_index = list(layer_count.index)
            layer_index.sort()
            max_gap = 1
            for id in range(len(layer_index) - 1):
                gap = layer_index[id + 1] - layer_index[id]
                if gap > 5:
                    max_gap = gap
                    break
            if (max_gap > 5):
                event_process = event_process.drop(event_process[event_process.trackIndex == trackId].index)
                drop_problem = True
                problem = problem + 'trackId: ' + str(trackId) + ' ' + 'gap_layer>5 ' + '\n'
                problem_count['gap_layer>5'] = problem_count['gap_layer>5'] + 1
                continue


            cross_layer = len(layer_index)
            if (cross_layer < 8):
                event_process.loc[(event_process[event_process.trackIndex == trackId].index), 'trackIndex'] = -1
                noise_problem = True
                problem = problem + 'trackId: ' + str(trackId) + ' ' + 'cross_layer<8' + '\n'
                problem_count['cross_layer<8'] = problem_count['cross_layer<8'] + 1
                continue
            '''
            pid_count = track.value_counts('currentTrackPID')
            if (len(pid_count) > 1):
                for idx, count in pid_count.iteritems():
                    if (count < 10):
                        noise_problem = True
                        event_process.loc[event_process[(event_process.trackIndex == trackId) & (
                                event_process.currentTrackPID == idx)].index, 'trackIndex'] = -1
            if (noise_problem):
                problem = problem + 'trackId: ' + str(trackId) + ' ' + 'pid_noise ' + '\n'
                problem_count['pid_noise'] = problem_count['pid_noise'] + 1
            creator_count_2 = pid_count[pid_count >= 10]

            if (len(creator_count_2) > 1):
                drop_problem = True
                #event_process = event_process.drop(event_process[event_process.event == event_id].index)
                problem = problem + 'trackId: ' + str(trackId) + ' ' + 'pid_drop ' + '\n'
                problem_count['pid_drop'] = problem_count['pid_drop'] + 1
            '''

            creator_count = track.value_counts('creatorProcessturnID')
            # pid_max_part = pid_count.max() / pid_count.sum()
            #if (creator_count.idxmax() != 'Decay'):
            #    event_process = event_process.drop(event_process[event_process.trackIndex == trackId].index)
            #    drop_problem = True
            #    problem = problem + 'trackId: ' + str(trackId) + ' ' + 'creator_drop ' + '\n'
            #    problem_count['creator_drop'] = problem_count['creator_drop'] + 1
            #    continue
            if (len(creator_count) > 1):
                for idx, count in creator_count.iteritems():
                    if (count < 10):
                        noise_problem = True
                        event_process.loc[event_process[(event_process.trackIndex == trackId) & (
                                    event_process.creatorProcessturnID == idx)].index, 'trackIndex'] = -1
            if (noise_problem):
                problem = problem + 'trackId: ' + str(trackId) + ' ' + 'creator_noise ' + '\n'
                problem_count['creator_noise'] = problem_count['creator_noise'] + 1

            creator_count_2 = creator_count[creator_count >= 10]
            if (len(creator_count_2) > 0 ):
                drop_index = event_process[(event_process.trackIndex == trackId) & (event_process.isScondary != 0)].index
                if ((len(drop_index)<(track.shape[0]-10)) and len(drop_index)>0):
                    drop_problem = True
                    event_process.loc[drop_index, 'trackIndex'] = -1
                    problem = problem + 'trackId: ' + str(trackId) + ' ' + 'creator_noise ' + '\n'
                    problem_count['creator_noise'] = problem_count['creator_noise'] + 1
                elif(len(drop_index)>(track.shape[0]-10)):
                    drop_problem = True
                    event_process = event_process.drop(drop_index)
                    problem = problem + 'trackId: ' + str(trackId) + ' ' + 'creator_drop ' + '\n'
                    problem_count['creator_drop'] = problem_count['creator_drop'] + 1

            #elif(len(creator_count_2) == 1):
            #    drop_index = event_process[(event_process.trackIndex ==trackId) & (event_process.isScondary ==1)].index
            #    if (len(drop_index) != track.shape[0]):
            #        #print('isSecondary:', event_id, trackId)
            #        drop_problem = True
            #        event_process = event_process.drop(drop_index)
            #        problem = problem + 'trackId: ' + str(trackId) + ' ' + 'creator_drop ' + '\n'
            #        problem_count['creator_drop'] = problem_count['creator_drop'] + 1


        #clean_data = clean_data.sort_value(by = 'gid')
        event_process = event_process.sort_values(by='gid')
        if (count==0):
            if (drop_problem):
                problem_event = event_process
                problem_event['problem'] = problem
                problem_event.to_csv(problem_out_file)
                #problem_data = pd.concat([problem_data, event_process])
            else:
                event_process.to_csv(clean_out_file)
        else:
            if (drop_problem):
                problem_event = event_process
                problem_event['problem'] = problem
                problem_event.to_csv(problem_out_file, mode='a', header=False)
                #problem_data = pd.concat([problem_data, event_process])
            else:
                event_process.to_csv(clean_out_file, mode='a',header=False)
        #clean_data = pd.concat([clean_data, event_process])
        
        count=count+1
        print(event_id, 'processed')
        if(count>args.event_max):
            break   
    print(problem_count)
    
    return 

def process_file(rawData_dir, fname, out_dir, wirePos_file, args):
    # load wire position and calculate (phi, r) from (x, y)
    t0 = time.time()
    data_file = os.path.join(rawData_dir, fname)

    wirePos = load_wirePos(wirePos_file)
    rawData = pd.read_csv(data_file, index_col=False, nrows=1000000)
    
    clean_out_file = os.path.join(out_dir, fname[:fname.find('.csv')] + '_cleaned' + fname[fname.find('.csv'):])
    problem_out_file = os.path.join(out_dir, fname[:fname.find('.csv')] + '_problem' + fname[fname.find('.csv'):])

    print("Processing #%d hits in %s." % (len(rawData), fname))

    t1 = time.time()
    #hits = allocate_wirePos(wirePos, rawData) 
    t2 = time.time()
    clean_data(rawData, wirePos, args, clean_out_file, problem_out_file)
    t3 = time.time()
    #clean_data = clean_data.sort_values(by='event')
    #problem_data = problem_data.sort_values(by='event')
    
    
    #clean_data.to_csv(output_file, index=False)
    #problem_data.to_csv(problem_file, index=False)

    t4 = time.time()
    print("Total time used: %d seconds, allocated wire position in %d seconds, cleaned data in %d seconds" % (t4- t0, t2-t1, t3-t3))



def main():
    parser = argparse.ArgumentParser(description = 'MDC data preprocess implementation')
    parser.add_argument('--rawDriftTime-max', type=float, default=1600)
    parser.add_argument('--event-max', type=float, default=1000)

    args = parser.parse_args()

    rawData_dir = "./data/rawData/rhoPi"
    out_dir = "./data/processedData/rhoPi"
    wirePos_file = "./data/preprocess/MdcWirePosition.csv"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for fname in os.listdir(rawData_dir):
        if fname.endswith(".csv"):
            print("Processing file: " + fname)
            process_file (rawData_dir, fname, out_dir, wirePos_file, args)

if __name__ == '__main__':
    main()
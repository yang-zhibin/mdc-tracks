import pandas as pd
import numpy as np

def polar_to_rc(hits):
    #original (x, y) is (r, phi)

    x = hits['x']*np.cos(hits['y'])
    y = hits['x']*np.sin(hits['y'])

    return x, y

def main():

    input_dir = 'E:\ihep\BESIII\hep_track\output_2\h2t_polar\h2t_polar_v3'
    hit_file = input_dir + '\hits_prediction_test.csv'
    output = input_dir + '\hits_prediction_test_rc.csv'

    hits = pd.read_csv(hit_file)
    x, y = polar_to_rc(hits)

    hits['x'] = y
    hits['y'] = x

    hits.to_csv(output)

if __name__ == '__main__':
    main()
import pandas as pd

def main():

    input_dir = './results/h2t_v1_2'
    hit_file = input_dir + '/'+'hits_prediction_test.csv'
    param_gnd_file = input_dir + '/'+'param_gnd_test.csv'
    param_pred_file = input_dir + '/'+'param_prediction_test.csv'

    hit = pd.read_csv(hit_file, index_col=0)
    param_gnd = pd.read_csv(param_gnd_file, index_col=0)
    param_pre = pd.read_csv(param_pred_file, index_col=0)
if __name__ == '__main__':
    main()
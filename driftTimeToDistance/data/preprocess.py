import pandas as pd
import matplotlib.pyplot as plt

def main():
    file = 'D:\ihep\mdc-tracks\driftTimeToDistance\data\pipijpsi_drift.csv'
    data = pd.read_csv(file)
    
    data = data.drop(data[data.driftDistance < 0].index)
    data = data.drop(data[data.driftDistance > 13].index)
    data = data.drop(data[data.rawDriftTime > 1500].index)
    
    num = 1500
    interval = 1500 /num
    
    fit_data = pd.DataFrame(columns=['rawDriftTime', 'driftDistance'])
    for i in range(num):
        print(i)
        lower = i*interval
        upper = (i+1)*interval
        gap = data.query('rawDriftTime > @lower and rawDriftTime < @upper')
        mean = gap.driftDistance.median()
        fit_data = fit_data.append({'rawDriftTime': (lower+upper)/2, 'driftDistance': mean}, ignore_index=True)
        #print(fit_data)
    
    fit_data.to_csv('driftTimeToDistance\data\\fit_drift_data.csv', index=False)    
    fit_data.plot(x='rawDriftTime', y='driftDistance', kind='scatter', figsize=(20, 16))
    plt.savefig('driftTimeToDistance\plots\\fit_drift_data.png')
    
if __name__ == "__main__":
    main()
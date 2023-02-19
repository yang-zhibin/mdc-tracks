import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    digi = pd.read_csv('E:\graph_ML\BESIII\data\data_processed\mdcDigi_1_0_eastPosandTrackId.csv')
    digi_grouped = digi.groupby('trackIndex')

    noise_digi = digi_grouped.get_group(-1)
    digi = digi.drop(digi[digi.trackIndex < 0].index)

    sns.set_theme(style='ticks')

    fig, axes = plt.subplots(1, 2)

    g1 = sns.jointplot(x=digi['east_x'], y=digi['east_y'], kind='hex', color='b')
    axes[0] = g1.fig.suptitle('Hits Distribution')
    g2 = sns.jointplot(x=noise_digi['east_x'], y=noise_digi['east_y'], kind='hex', color='orange')
    axes[1] = g2.fig.suptitle('Noise Distribution')


    max_event = digi['event'].max()
    title = "Hits and Noise Distribution, count events: " + str(max_event)
    #fig.suptitle(title)
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


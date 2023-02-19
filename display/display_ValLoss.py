import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def draw_result(result, output, pdf):

    epoch = result['epoch']
    for index, col in result.iteritems():
        if (index == 'epoch'):
            continue

        plt.plot(epoch, col, 'o-')
        plt.xlabel('epoch')
        plt.ylabel(index)
        plt.title(index)
        pdf.savefig()
        plt.close()


def main():

    input_dir = 'E:\ihep\BESIII\hep_track\output_2\h2t_v4\hit2track_baseline_v3\loss'
    file_name = '\ValLoss'

    input_file = input_dir + file_name + '.csv'
    output = 'h2t_v4_valLoss.pdf'

    result = pd.read_csv(input_file)

    with PdfPages(output) as pdf:
        draw_result(result, output, pdf)
        print('Save to pdf file',pdf)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


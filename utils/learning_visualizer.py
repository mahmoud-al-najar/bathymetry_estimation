import matplotlib.pyplot as plt
import numpy as np
import csv


def visualize_learning(log_file_path, measures_to_plot=None, fig_save_name=None):
    if measures_to_plot is None:
        measures_to_plot = ['loss', 'val_loss']
    with open(log_file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        rows = list(reader)

        headers = rows[0]
        values = np.array(rows[1:]).astype(float)

        best_epoch_index = 200
        print(values[best_epoch_index])
        indices = []
        for m in measures_to_plot:
            indices.append(headers.index(m))

        plt.figure()
        legend = []
        for i in indices:
            plt.plot(values[:, i])
            legend.append(headers[i])
        plt.legend(legend)
        plt.xlabel("Epoch #")
        plt.ylabel('Losses')

        # plt.annotate('axes fraction',
        #             xy=(4.86833236e-03, best_epoch_index), xycoords='data',
        #             textcoords='axes fraction',
        #             arrowprops=dict(facecolor='black', shrink=0.05),
        #             horizontalalignment='right', verticalalignment='top')

        if fig_save_name is not None:
            plt.savefig(fig_save_name)
        else:
            plt.show()


visualize_learning('C:/Users/Al-Najar/Desktop/full_training.csv')

import matplotlib.pyplot as plt
import numpy as np
import csv


def visualize_learning(log_file_path, measures_to_plot=None, fig_save_name=None):
    if measures_to_plot is None:
        measures_to_plot = ['loss', 'val_loss']
    with open(log_file_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        rows = list(reader)

        headers = rows[0]
        values = np.array(rows[1:]).astype(float)

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
        if fig_save_name is not None:
            plt.savefig(fig_save_name)
        else:
            plt.show()


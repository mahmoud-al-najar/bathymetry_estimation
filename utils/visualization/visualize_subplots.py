import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from math import sqrt


params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 150,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'font.size': 8,
    'legend.fontsize': 6,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'font.family': 'serif',
}
matplotlib.rcParams.update(params)


def visualize(real_bathy, predicted_bathy, title, show_3d=False, output_path=None, show=True):
    # Prepare data
    padding = 0
    if real_bathy.shape[0] != predicted_bathy.shape[0]:
        padding += int((real_bathy.shape[0] - predicted_bathy.shape[0]) / 2)
        predicted_bathy = np.pad(predicted_bathy, padding, mode='constant')

    padding += 1  # ######################################################### TEMP - to cut one more pixel off the edges

    real_without_padding = real_bathy[padding:real_bathy.shape[0]-padding, padding:real_bathy.shape[1]-padding]
    pred_without_padding = predicted_bathy[
                           padding:predicted_bathy.shape[0]-padding,
                           padding:predicted_bathy.shape[1]-padding]

    avg_real_without_padding = np.average(real_without_padding, axis=0)
    avg_pred_without_padding = np.average(pred_without_padding, axis=0)

    # Calculate RMSE's
    full_matrix_rmse = round(sqrt(mean_squared_error(real_without_padding, pred_without_padding)), 2)
    average_profile_rmse = round(sqrt(mean_squared_error(avg_real_without_padding, avg_pred_without_padding)), 2)

    vmin = np.min(-real_bathy) if np.min(-real_bathy) < np.min(-predicted_bathy) else np.min(-predicted_bathy)
    vmax = np.max(-real_bathy) if np.max(-real_bathy) > np.max(-predicted_bathy) else np.max(-predicted_bathy)

    # Visualise in 3D
    if show_3d:
        x = np.arange(0, real_bathy.shape[0], 1)
        y = np.arange(0, real_bathy.shape[1], 1)
        X, Y = np.meshgrid(x, y)
        fig = plt.figure(figsize=(15, 10), dpi=100)
        ax = fig.gca(projection='3d')
        ax.set_title(title)
        im = ax.plot_surface(X, Y, -real_bathy, cmap='Blues', linewidth=0.5, vmin=vmin, vmax=vmax)
        ax.plot_surface(X, Y, -predicted_bathy, cmap='Reds', linewidth=0.5, vmin=vmin, vmax=vmax)
        ax.view_init(elev=18, azim=92)
        ax.grid(b=False)
        plt.colorbar(im, ax=ax, shrink=0.5)

    # Visualise in 2D subplots
    fig3, (ax3_1, ax3_2, ax3_3) = plt.subplots(1, 3, figsize=(11 * 1.3, 3 * 1.3))
    ax3_1.imshow(-real_without_padding, cmap='ocean_r', vmin=vmin, vmax=vmax, aspect='equal')
    im3_2 = ax3_2.imshow(-pred_without_padding, cmap='ocean_r', vmin=vmin, vmax=vmax, aspect='equal')
    divider = make_axes_locatable(ax3_2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3_2, cax=cax)
    ax3_1.set_title(title + ' - target bathymetry')
    ax3_2.set_title(title + ' - estimated bathymetry\nrmse = ' + str(full_matrix_rmse) + ' meters')

    ax3_3.plot(-avg_real_without_padding)
    ax3_3.plot(-avg_pred_without_padding)
    ax3_3.set_title(title + ' - estimated average profile\nrmse = ' + str(average_profile_rmse) + ' meters')

    # Save
    if output_path is not None:
        plt.savefig(output_path + f'{title}_reconstructed_2d_gradient__removed_edge')

    # Show
    if show:
        plt.show()

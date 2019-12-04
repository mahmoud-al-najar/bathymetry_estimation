import scipy.io as sio
import numpy as np
from numpy import inf
import math
import json
import netCDF4
import cmocean
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.models import load_model
from sklearn.metrics import mean_squared_error


def myround(x, base=5):
    return base * round(x/base)


def find_nearest(array, value):
    array = np.asarray(array)
    array = np.average(array, axis=0)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def interval_rmse(target, result, s, e):
    return math.sqrt(mean_squared_error(target[:, s:e], result[:, s:e]))


plt.rcParams.update({'font.size': 14})
model_path = r'C:\Users\Al-Najar\Desktop\SGD_fulldataset_msle-weights-improvement-140-0.01.hdf5'
model = load_model(model_path)

names = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
interval = 10
errors = dict()
max_depth = 0
for fname in names:
    mat_file = sio.loadmat('physics/' + fname + '.mat')
    # x_axis = mat_file['bathy'][0, 0]['xm'][:, 0]
    depth = mat_file['bathy'][0, 0]['kDep'][0, 0]['D']
    depth = np.array(depth).astype(float)
    where_are_NaNs = np.isnan(depth)
    depth[where_are_NaNs] = 0
    depth[depth == inf] = 0
    depth[depth == -inf] = 0
    dxm = mat_file['bathy'][0, 0]['dxm'][0, 0]
    dym = mat_file['bathy'][0, 0]['dym'][0, 0]

    filename = 'output_' + fname
    base_path = r'C:\Users\Al-Najar\Downloads\alloutput\\' + filename
    nc_bathy = base_path + r'\dep.nc'
    nc_waves = base_path + r'\eta.nc'

    waves_sim = netCDF4.Dataset(nc_waves)['eta']
    bst = waves_sim[52:56]

    burst = np.rollaxis(bst, 0, 3)
    burst = (burst - np.min(burst[:, :100, :])) / np.max(burst[:, :100, :])
    burst = np.expand_dims(burst, axis=0)
    res = model.predict(burst)[0, :, :, 0] * 10
    res = res[::dxm, ::dym]

    bathy = netCDF4.Dataset(nc_bathy)['depth']
    bathy = np.array(bathy).astype(float)
    bathy = bathy[::dxm, ::dym]
    if np.max(bathy) > max_depth:
        max_depth = np.max(bathy)
    depth_ranges = np.arange(0, max_depth, interval)

    for v in depth_ranges:
        start_depth = v
        end_depth = start_depth + interval - 1
        dl_key_string = 'dl_' + str(start_depth) + 'm - ' + str(end_depth) + 'm'
        pm_key_string = 'pm_' + str(start_depth) + 'm - ' + str(end_depth) + 'm'
        if dl_key_string not in errors.keys():
            errors[dl_key_string] = []
            errors[pm_key_string] = []

    for v in depth_ranges:
        start_depth = v
        end_depth = start_depth + interval - 1
        dl_key_string = 'dl_' + str(start_depth) + 'm - ' + str(end_depth) + 'm'
        pm_key_string = 'pm_' + str(start_depth) + 'm - ' + str(end_depth) + 'm'
        if end_depth <= np.max(bathy):
            start = find_nearest(bathy, end_depth)[0]
            end = find_nearest(bathy, start_depth)[0]
            dl_rmse = interval_rmse(bathy, res, start, end)
            pm_rmse = interval_rmse(bathy, depth, start, end)
            errors[dl_key_string].append(dl_rmse)
            errors[pm_key_string].append(pm_rmse)

# data_a = [[1, 2, 5], [5, 7, 2, 2, 5], [7, 2, 5]]
# data_b = [[6,4,2], [1,2,5,3, 2], [2, 3, 5, 1]]
data_a = []
data_b = []
dl_line_data = []
pm_line_data = []
ticks = []
for v in depth_ranges[:-1]:
    start_depth = v
    end_depth = start_depth + interval - 1
    dl_key_string = 'dl_' + str(start_depth) + 'm - ' + str(end_depth) + 'm'
    pm_key_string = 'pm_' + str(start_depth) + 'm - ' + str(end_depth) + 'm'
    data_a.append(errors[dl_key_string])
    data_b.append(errors[pm_key_string])
    dl_line_data.append(np.average(errors[dl_key_string]))
    pm_line_data.append(np.average(errors[pm_key_string]))
    ticks.append(dl_key_string.replace('dl_', ''))

# ticks = range(len(data_b))



def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['means'], color=color)


plt.figure(figsize=(15, 10))

bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6, showmeans=True, meanline=True)
bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6, showmeans=True, meanline=True)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

plt.xticks(range(0, len(ticks) * 2, 2), ticks, rotation=45)
dl_line_xs = []
for x in bpl['means']:
    dl_line_xs.append(np.average(x.get_xdata()))
pm_line_xs = []
for x in bpr['means']:
    pm_line_xs.append(np.average(x.get_xdata()))
print(dl_line_xs)
# draw temporary red and blue lines and use them to create a legend
# plt.plot([], c='#D7191C', label='Deep learning model')
# plt.plot([], c='#2C7BB6', label='Physics-based method')
plt.plot(pm_line_xs, pm_line_data, c='#2C7BB6', label='Physics-based method')
plt.plot(dl_line_xs, dl_line_data, c='#D7191C', label='Deep learning model')
plt.legend()

# plt.xlim(-2, len(ticks)*2)
# plt.ylim(0, 8)
plt.xlabel('Depth intervals [m]')
plt.ylabel('RMSE error')
plt.tight_layout()
plt.savefig('plots/errors_depth_intervals.png')
# plt.show()


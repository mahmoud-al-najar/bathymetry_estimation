import numpy as np
from keras.models import load_model
from utils.preprocessing import apply_2d_gradient, apply_normxcorr2, apply_per_band_minmax_normalization
from utils.reconstruction import reconstruct_tile
import copy
import matplotlib.pyplot as plt
import netCDF4
import time
from utils.visualization.visualize_subplots import visualize

# model = load_model('F:/models/sequential_1_25000_97_0.15.hdf5')
model = load_model('F:/models2/sequential_1_25000_36_0.08.hdf5')


# Makeshift function to predict and visualize a list of files
# To be modified depending on input formats
# Data passed to reconstruct_tile() should be in channels-last format
def temp(name):
    file_path = 'F:/synthetic_data/alloutput/'
    case_name = name
    directory = f'output_{case_name}/'
    waves_sim_fname = f'{directory}eta.nc'
    output_dir = 'F:/models2/results/'
    output_path = f'{output_dir + case_name}_reconstructed_2d_gradient.npy'

    t0 = time.time()
    waves_sim = netCDF4.Dataset(file_path + waves_sim_fname)['eta']
    waves_sim = waves_sim[-10:-6]
    waves_sim = np.rollaxis(waves_sim, 0, 3)

    result = reconstruct_tile(model, waves_sim, 40, apply_2d_gradient, apply_per_band_minmax_normalization)
    np.save(output_path, result)
    t1 = time.time()
    print(f'Time: {t1 - t0}')

    depth_fname = f'{file_path + directory}dep.nc'
    real_bathy = netCDF4.Dataset(depth_fname)['depth']
    real_bathy = np.array(real_bathy).astype(float)
    predicted_bathy = np.load(output_path)
    visualize(real_bathy, predicted_bathy, title=case_name, output_path=output_dir, show=False)


names = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
for n in names:
    temp(n)

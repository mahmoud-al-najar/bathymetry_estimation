import numpy as np
from keras.models import load_model
from utils.preprocessing import apply_fft, apply_normxcorr2, apply_per_band_minmax_normalization
from utils.reconstruction import reconstruct_tile
import copy
import matplotlib.pyplot as plt

# model = load_model('F:/models/sequential_1_25000_97_0.15.hdf5')
model = load_model('F:/models2/sequential_1_25000_36_0.08.hdf5')

file_path = 'C:/Users/Al-Najar/PycharmProjects/bathymetry_estimation/models/' \
            'to_copy/v10_perband-slopes/saint-louis/'
saint_louis_fname = 'tile20181204_SaintLouis_filtered.npy'
output_path = ''
saint_louis_tile = np.load(file_path + saint_louis_fname)

temp = copy.deepcopy(saint_louis_tile)
temp[:, :, 0] = saint_louis_tile[:, :, 0]
temp[:, :, 1] = saint_louis_tile[:, :, 3]
temp[:, :, 2] = saint_louis_tile[:, :, 1]
temp[:, :, 3] = saint_louis_tile[:, :, 2]
saint_louis_tile = temp

result = reconstruct_tile(model, saint_louis_tile, 40, apply_fft, apply_normxcorr2, apply_per_band_minmax_normalization)
np.save(output_path + saint_louis_fname.replace('.npy', 'reconstructed'), result)

a = np.load(output_path + saint_louis_fname.replace('.npy', 'reconstructed.npy'))
plt.imshow(a)
plt.colorbar()
plt.show()

import numpy as np
import re
import os
import csv
from utils.preprocessing import apply_normxcorr2, apply_2d_gradient


version = 14
raw_path = '/projets/reva/dwilson/bathymetry_estimation/new_dataset/untarred/'
out_path = f'/projets/reva/dwilson/bathymetry_estimation/new_dataset/processed/{version}/'
n_bursts = 50
tile_size = 40
n_random_bursts = 6
n_cross_shore_intervals = 3
n_cross_shore_interval_random_extracts = 4
target_dataset_size = 100_000

# extract_path, average_depth, hmo, freqpeak, thetapeak
out_csv = []

for dir_path, sub_dirs, files in os.walk(raw_path):
    for f in files:
        if f.startswith('b') and f.endswith('.npy'):

            # Prepare input paths
            n = f.replace('b', '').replace('.npy', '')
            path_b = dir_path + '/' + f
            path_w = dir_path + '/' + 'w' + n + '.npy'
            path_l = dir_path + '/' + 'LOG.txt_' + n

            # Prepare output directories
            res_n = dir_path.split('/')[len(dir_path.split('/')) - 2]
            res_out_dir = out_path + res_n + '/'
            if not os.path.exists(res_out_dir):
                os.mkdir(res_out_dir)
            sub_res_out_dir = res_out_dir + 'sim_' + n + '/'
            if not os.path.exists(sub_res_out_dir):
                os.mkdir(sub_res_out_dir)
            bursts_out_dir = sub_res_out_dir + 'bursts_' + n + '/'
            if not os.path.exists(bursts_out_dir):
                os.mkdir(bursts_out_dir)

            # Read input files
            bathy = np.load(path_b)
            waves = np.load(path_w)
            f = open(path_l, 'r')
            log_lines = f.readlines()
            f.close()

            # Remove simulation margins, spin-up frames
            bathy = bathy[60:-60, 50:]
            waves = waves[600:, 60:-60, 50:]

            normalized_elevation = waves

            # Calculate wave-breaking area
            hmo_line = log_lines[48]
            hmo = float(re.findall("-?\d+\.\d+", hmo_line)[0])
            wave_breaks = hmo / bathy
            bools = wave_breaks > 0.4
            indices = np.where(bools == True)
            cut_point = np.min(indices[1])

            # Extract wave features
            freqpeak_line = log_lines[45]
            freqpeak = float(re.findall("-?\d+\.\d+", freqpeak_line)[0])
            thetapeak_line = log_lines[50]
            thetapeak = float(re.findall("-?\d+\.\d+", thetapeak_line)[0])

            # Crop wave-breaking area
            normalized_and_cropped_elevation = normalized_elevation[:, :, :cut_point]
            cropped_bathy = bathy[:, :cut_point]

            # Pick n_random_bursts from all possible bursts
            # To get all bursts, modify to: for bst in range(n_bursts)
            for bst in np.random.choice(range(n_bursts), n_random_bursts):
                # Prepare sub-tiles output directory
                extracts_out_dir = bursts_out_dir + 'extracts_' + str(bst) + '/'
                if not os.path.exists(extracts_out_dir):
                    os.mkdir(extracts_out_dir)

                # Extract burst
                norm_burst = normalized_and_cropped_elevation[(22 * bst): (22 * bst) + 10]
                long_shore = norm_burst.shape[1]
                cross_shore = norm_burst.shape[2]

                # Loop through cross_shore_intervals
                cross_shore_interval_size = cross_shore / n_cross_shore_intervals
                for interval_start in [i * cross_shore_interval_size for i in range(n_cross_shore_intervals)]:
                    interval_end = interval_start + cross_shore_interval_size
                    for i in range(n_cross_shore_interval_random_extracts):
                        if interval_end - tile_size > interval_start:
                            y = np.random.choice(range(int(interval_start), int(interval_end - tile_size - 1)))
                            x = np.random.choice(range(0, long_shore - tile_size - 1))
                            avg_bathy = np.average(cropped_bathy[x:x + tile_size, y:y + tile_size])
                            sub_elevation = norm_burst[:, x:x + tile_size + 1, y:y + tile_size + 1]

                            sub_elevation = apply_2d_gradient(sub_elevation)
                            sub_elevation = apply_normxcorr2(sub_elevation)

                            for i in range(sub_elevation.shape[0]):
                                sub_elevation[i] = (sub_elevation[i] - np.min(sub_elevation[i])) / \
                                                   (np.max(sub_elevation[i]) - np.min(sub_elevation[i]))

                            fname_extract = extracts_out_dir + 'extract_x' + str(x) + '_y' + str(y)
                            np.save(fname_extract, sub_elevation, allow_pickle=True)
                            out = (fname_extract, avg_bathy, hmo, freqpeak, thetapeak)
                            out_csv.append(out)
            if len(out_csv) > target_dataset_size:
                with open(f'v{version}.csv', 'w', newline='') as csv_file:
                    writer2 = csv.writer(csv_file, delimiter=',')
                    writer2.writerows(out_csv)
                print('DONE')
                exit()

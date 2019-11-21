from keras.models import load_model
import csv
import math
import numpy as np
import os
from keras import backend as K
from keras import losses
from keras import metrics
from sklearn.metrics import mean_squared_error


def max_error(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred))


metrics.max_error = max_error
losses.max_error = max_error
with open('val_ids.csv', 'r') as f:  # TO BE SET: ids csv
    reader = csv.reader(f)
    all_bursts_and_bathymetries = list(reader)

full_results = []
headers = ('model', 'rmse', 'mse', 'rmse_middle', 'mse_middle', 'avg_slope_error', 'rmse_shallow')
full_results.append(headers)
count = 0

dir_path = ''  # TO BE SET: directory with trained auto-encoders '.hdf5'
for f in os.listdir(dir_path):
    if f.endswith('.hdf5'):
        model_name = f
        full_model_path = dir_path + model_name
        model = load_model(full_model_path)
        count += 1
        print(str(count) + ': ' + full_model_path)

        list_rmse = []
        list_mse = []

        list_rmse_middle = []
        list_mse_middle = []
        list_avg_slope_error = []

        list_rmse_shallow = []

        for row in all_bursts_and_bathymetries:
            test_bathy = np.load(row[0])
            test_bst = np.load(row[1])[:4, :, :]
            test_bst = np.rollaxis(test_bst, 0, 3)
            test_bst = np.expand_dims(test_bst, axis=0)

            res = model.predict(test_bst)
            res = res[0, :, :, 0]
            res = res * 10

            list_rmse.append(math.sqrt(mean_squared_error(test_bathy, res)))
            list_mse.append(mean_squared_error(test_bathy, res))

            list_rmse_middle.append(math.sqrt(mean_squared_error(test_bathy[:, 30:-30], res[:, 30:-30])))
            list_mse_middle.append(mean_squared_error(test_bathy[:, 30:-30], res[:, 30:-30]))

            tb_slope = np.average(np.gradient(np.average(test_bathy, axis=0)))
            res_slope = np.average(np.gradient(np.average(res, axis=0)))
            slope_error = np.abs(tb_slope) - np.abs(res_slope)
            slope_error = np.abs(slope_error)
            list_avg_slope_error.append(slope_error)

            index_shallow = int(35/tb_slope/10)  # divide by 10 because the data is downscaled (10m resolution)
            list_rmse_shallow.append(
                math.sqrt(mean_squared_error(test_bathy[:, :index_shallow], res[:, :index_shallow])))

        rmse = np.average(list_rmse)
        mse = np.average(list_mse)
        avg_slope_error = np.average(list_avg_slope_error)

        rmse_middle = np.average(list_rmse_middle)
        mse_middle = np.average(list_mse_middle)

        rmse_shallow = np.average(list_rmse_shallow)
        results_json = {
            'model_path': full_model_path,
            'rmse': float(rmse),
            'mse': float(mse),
            'rmse_middle': float(rmse_middle),
            'mse_middle': float(mse_middle),
            'avg_slope_error': float(avg_slope_error),
            'rmse_shallow': float(rmse_shallow)
        }

        full_results.append(tuple(results_json.values()))

with open('results.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(full_results)

print('DONE')

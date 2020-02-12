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
# headers = ('model', 'rmse', 'mse', 'rmse_middle', 'mse_middle', 'avg_slope_error', 'rmse_shallow')


def find_nearest(array, value):
    array = np.asarray(array)
    array = np.average(array, axis=0)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def evaluate_single_case(model_path, burst, bathy):
    """
    Arguments:
        model_path: path to auto-encoder
        burst: numpy wave burst array. shape: (1, 200, 200, 4)
        bathy: numpy bathymetry profile array. shape: (200, 200)

    Returns:
        dict{model_path, rmse, mse, rmse_middle, mse_middle, slope_error, rmse_shallow}
    """
    model = load_model(model_path)
    res = model.predict(burst)
    res = res[0, :, :, 0]
    res = res * 10

    rmse = math.sqrt(mean_squared_error(bathy, res))
    mse = mean_squared_error(bathy, res)

    rmse_middle = math.sqrt(mean_squared_error(bathy[:, 30:-30], res[:, 30:-30]))
    mse_middle = mean_squared_error(bathy[:, 30:-30], res[:, 30:-30])

    tb_slope = np.average(np.gradient(np.average(bathy, axis=0)))
    res_slope = np.average(np.gradient(np.average(res, axis=0)))
    slope_error = np.abs(tb_slope) - np.abs(res_slope)
    slope_error = np.abs(slope_error)

    index_shallow = int(35 / tb_slope / 10)  # divide by 10 because the data is downscaled (10m resolution)
    rmse_shallow = math.sqrt(mean_squared_error(bathy[:, :index_shallow], res[:, :index_shallow]))

    index_0m = np.argmax(np.bincount(np.where(bathy > 0, bathy, np.inf).argmin(axis=1)))
    index_40m, _ = find_nearest(bathy, 40)
    rmse_between_0_and_40 = math.sqrt(mean_squared_error(bathy[:, index_40m:index_0m], res[:, index_40m:index_0m]))

    results_dict = {
        'model_path': model_path,
        'rmse': float(rmse),
        'mse': float(mse),
        'rmse_middle': float(rmse_middle),
        'mse_middle': float(mse_middle),
        'slope_error': float(slope_error),
        'rmse_shallow': float(rmse_shallow),
        'rmse_between_0_and_40': float(rmse_between_0_and_40),
        'index_0m': int(index_0m),
        'index_40m': int(index_40m)
    }

    return results_dict

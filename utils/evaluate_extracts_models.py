from keras.models import load_model
import csv
import numpy as np
import os


with open('extracts.csv', 'r') as f:
    reader = csv.reader(f)
    extracts = list(reader)

models_errors = []
dir_path = ''  # TO BE SET -- directory of trained models
for d in os.listdir(dir_path):
    model_dir = dir_path + d + '/'
    for f in os.listdir(model_dir):
        if f.endswith('hdf5'):
            model_name = f
            full_model_path = model_dir + model_name
            model = load_model(full_model_path)

            results = []
            error = []
            for row in extracts:
                test_bst = np.load(row[0] + '.npy')
                test_bst = np.rollaxis(test_bst, 0, 3)
                test_bst = np.expand_dims(test_bst, axis=0)
                test_bst = test_bst[:, :, :, :4]
                res = model.predict(test_bst)
                res = res * 10
                error.append((float(row[1])-res))
                results.append(float(res[0][0]))

            mse = np.mean(np.array([i**2 for i in error]))
            max_error = np.max(np.abs(np.array(error)))
            std_dev = np.std(np.array(results))
            models_errors.append((d, mse, max_error, std_dev))
            param_string = d
            param_parts = d.split('___')
            epsilon = float(param_parts[0].split('_')[1])
            beta1 = float(param_parts[1].split('_')[1])
            beta2 = float(param_parts[2].split('_')[1])

            with open('results_incremental_FINAL.csv', 'a') as res_file:
                res_file.write(d + ',' + str(epsilon) + ',' + str(beta1) + ',' + str(beta2) + ',' + str(mse) + ','
                               + str(max_error) + ',' + str(std_dev) + '\n')
            with open('test.log', 'a') as log:
                log.write(str(len(models_errors)) + ': ' + d + '\n')
            print(str(len(models_errors)) + ': ' + d)

with open('all_results_FINAL.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(models_errors)


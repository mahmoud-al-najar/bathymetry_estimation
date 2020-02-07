import csv
import random
from full_area_estimation.bathymetry.data.generator import DataGenerator
from full_area_estimation.bathymetry.models.unet import UNet
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
import math
from tensorflow import set_random_seed
import numpy as np
from keras import backend as K

def max_error(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred))

with open('ids.csv', 'r') as f:
    reader = csv.reader(f)
    all_bursts_and_bathymetries = list(reader)

seed = 0
set_random_seed(seed)
random.seed(seed)
random.shuffle(all_bursts_and_bathymetries)

#dataset_size = 200000
list_ids = []
labels = dict()

for r in all_bursts_and_bathymetries:
    list_ids.append(r[0])
    labels[r[0]] = r[1]

train_test_ratio = 0.9
validation_split = 0.2
train_size = math.floor(train_test_ratio * len(list_ids))
validation_size = train_size * 0.2

partition = dict()
partition['train'] = list_ids[:int(train_size-validation_size)]
partition['validation'] = list_ids[int(train_size-validation_size):int(train_size)]
partition['test'] = list_ids[int(train_size):]
print('Training: ' + str(len(partition['train'])))
print('Validation: ' + str(len(partition['validation'])))
print('Test: ' + str(len(partition['test'])))

generator_train = DataGenerator(partition['train'], labels)
generator_test = DataGenerator(partition['test'], labels)
generator_validation = DataGenerator(partition['validation'], labels)

unet = UNet()
model = unet.create_model()
model.summary()

total_items = len(generator_train)
batch_size = 256
epochs = 200
num_batches = int(total_items/batch_size)

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
csv_logger = CSVLogger('training.log', separator=',')
filepath="SGD_fulldataset_mse-weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
cp = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='min', period=5)

# mean_squared_logarithmic_error
# mean_squared_error
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_logarithmic_error', 'mean_squared_error'])
history = model.fit_generator(generator=generator_train, steps_per_epoch=num_batches, epochs=epochs, verbose=1, validation_data=generator_validation, callbacks=[csv_logger,cp])

scores = model.evaluate_generator(generator=generator_test)

model.save('sgd_mse_model_fulldataset')
print(scores)

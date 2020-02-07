import csv
import random
from full_area_estimation.bathymetry.data.generator50x50 import DataGenerator
from full_area_estimation.bathymetry.models.unet50x50 import UNet
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
import math
from tensorflow import set_random_seed

with open('ids.csv', 'r') as f:
    reader = csv.reader(f)
    all_bursts_and_bathymetries = list(reader)

seed = 0
set_random_seed(seed)
random.seed(seed)
random.shuffle(all_bursts_and_bathymetries)

dataset_size = 100000
list_ids = []
labels = dict()

for r in all_bursts_and_bathymetries[:dataset_size]:
    for i in range(4):
        id = r[0] + '__' + str(i)
        label = r[1] + '__' + str(i)
        list_ids.append(id)
        labels[r[0]] = label

random.shuffle(list_ids)

train_test_ratio = 0.9
validation_split = 0.2
train_size = math.floor(train_test_ratio * len(list_ids))
validation_size = train_size * 0.2

partition = dict()
partition['train'] = list_ids[:int(train_size - validation_size)]
partition['validation'] = list_ids[int(train_size - validation_size):int(train_size)]
partition['test'] = list_ids[int(train_size):]
print('Training: ' + str(len(partition['train'])))
print('Validation: ' + str(len(partition['validation'])))
print('Test: ' + str(len(partition['test'])))

batch_size = 64
epochs = 100

generator_train = DataGenerator(partition['train'], labels, batch_size=1,
                                x_shape=(50, 50, 4), y_shape=(50, 50, 1))
generator_test = DataGenerator(partition['test'], labels, batch_size=1,
                               x_shape=(50, 50, 4), y_shape=(50, 50, 1))
generator_validation = DataGenerator(partition['validation'], labels, batch_size=1,
                                     x_shape=(50, 50, 4), y_shape=(50, 50, 1))

unet = UNet()
model = unet.create_model()
model.summary()

total_items = len(generator_train)
batch_size = 64
epochs = 200
num_batches = int(total_items/batch_size)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
csv_logger = CSVLogger('training.log', separator=',')
filepath="50x50-Adam-weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
cp = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# mean_squared_logarithmic_error
# mean_squared_error
model.compile(loss='mean_squared_logarithmic_error', optimizer=Adam(), metrics=['mean_squared_logarithmic_error', 'mean_squared_error'])
history = model.fit_generator(generator=generator_train, steps_per_epoch=num_batches, epochs=epochs, verbose=1,
                              validation_data=generator_validation, callbacks=[csv_logger, es, cp])

scores = model.evaluate_generator(generator=generator_test)

model.save('50x50_Adam_model')
print(scores)

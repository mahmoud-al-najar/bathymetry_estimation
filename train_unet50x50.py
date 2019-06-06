import csv
import random
from bathymetry.data.generator50x50 import DataGenerator
from bathymetry.models.unet50x50 import UNet
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, CSVLogger
import math

with open('ids.csv', 'r') as f:
    reader = csv.reader(f)
    all_bursts_and_bathymetries = list(reader)

random.seed(449)
random.shuffle(all_bursts_and_bathymetries)

dataset_size = 50000
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

generator_train = DataGenerator(partition['train'], labels, batch_size=batch_size,
                                x_shape=(50, 50, 4), y_shape=(50, 50, 1))
generator_test = DataGenerator(partition['test'], labels, batch_size=batch_size,
                               x_shape=(50, 50, 4), y_shape=(50, 50, 1))
generator_validation = DataGenerator(partition['validation'], labels, batch_size=batch_size,
                                     x_shape=(50, 50, 4), y_shape=(50, 50, 1))

unet = UNet()
model = unet.create_model()
model.summary()

total_items = len(generator_train)
num_batches = int(total_items / batch_size)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
csv_logger = CSVLogger('training.log', separator=',')
model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mean_squared_error'])

# from keras.utils import plot_model
# plot_model(model, to_file='unet50x50.png', show_shapes=True, show_layer_names=True)

history = model.fit_generator(generator=generator_train, steps_per_epoch=num_batches, epochs=epochs, verbose=1,
                              validation_data=generator_validation, callbacks=[es, csv_logger])

f_val_loss = open("val_loss.txt", "w+")
for h in history.history['val_loss']:
    f_val_loss.write(str(h) + '\n')

f_loss = open("loss.txt", "w+")
for h in history.history['loss']:
    f_loss.write(str(h) + '\n')

scores = model.evaluate_generator(generator=generator_test)

model.save('sgd_model')
print(scores)

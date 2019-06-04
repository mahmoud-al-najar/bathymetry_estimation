import csv
import random
from bathymetry.data.generator import DataGenerator
from bathymetry.models.unet import UNet
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, CSVLogger
import math


with open('ids.csv', 'r') as f:
    reader = csv.reader(f)
    all_bursts_and_bathymetries = list(reader)

random.seed(449)
random.shuffle(all_bursts_and_bathymetries)

list_ids = []
labels = dict()

for r in all_bursts_and_bathymetries:
    list_ids.append(r[0])
    labels[r[0]] = r[1]

train_test_ratio = 0.9
train_size = math.floor(train_test_ratio * len(all_bursts_and_bathymetries))

partition = dict()
partition['train'] = list_ids[:train_size]
partition['test'] = list_ids[train_size:]

generator_train = DataGenerator(partition['train'], labels)
generator_test = DataGenerator(partition['test'], labels)

unet = UNet()
model = unet.create_model()
model.summary()

total_items = len(generator_train)
batch_size = 64
epochs = 100
num_batches = int(total_items/batch_size)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
csv_logger = CSVLogger('training.log', separator=',')
model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mean_squared_error'])
history = model.fit_generator(generator=generator_train, steps_per_epoch=num_batches, epochs=epochs, verbose=1,
                              validation_split=0.2, callbacks=[es, csv_logger])

f_val_loss = open("val_loss.txt", "w+")
for h in history.history['val_loss']:
    f_val_loss.write(str(h) + '\n')

f_loss = open("loss.txt", "w+")
for h in history.history['loss']:
    f_loss.write(str(h) + '\n')

scores = model.evaluate_generator(generator=generator_test)

model.save('sgd_model')
print(scores)

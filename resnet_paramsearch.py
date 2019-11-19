from __future__ import print_function
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras.callbacks import ReduceLROnPlateau
from bathymetry.models.resnet import ResNet
import numpy as np
import random
from tensorflow import set_random_seed
import csv
import math
from bathymetry.data.extracts_generator import ExtractsGenerator
from keras import backend as K
import os

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 100:
        lr *= 0.5e-3
    elif epoch > 70:
        lr *= 1e-3
    elif epoch > 50:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# Setting up data generators
with open('extracts.csv', 'r') as f:
    reader = csv.reader(f)
    dataset = list(reader)

seed = 1
set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)
random.shuffle(dataset)

dataset_size = 25000
list_ids = []
labels = dict()

for r in dataset[:dataset_size]:
    list_ids.append(r[0])
    labels[r[0]] = r[1]

train_test_ratio = 0.85
validation_split = 0.28
train_size = math.floor(train_test_ratio * len(list_ids))
validation_size = train_size * validation_split

partition = dict()
partition['train'] = list_ids[:int(train_size-validation_size)]
partition['validation'] = list_ids[int(train_size-validation_size):int(train_size)]
partition['test'] = list_ids[int(train_size):]
print('Training: ' + str(len(partition['train'])))
print('Validation: ' + str(len(partition['validation'])))
print('Test: ' + str(len(partition['test'])))

generator_train = ExtractsGenerator(partition['train'], labels)
generator_test = ExtractsGenerator(partition['test'], labels)
generator_validation = ExtractsGenerator(partition['validation'], labels)

# Setting model parameters
n = 6
depth = n * 9 + 2
model_type = 'ResNet%d' % depth
input_shape = (20, 20, 4)
print(model_type)

total_items = len(partition['train'])
batch_size = 128
epochs = 60
num_batches = int(total_items/batch_size)

beta1_list = [0.1, 0.5, 0.9, 0.99, 0.999]  # 7
beta2_list = [0.999, 0.1, 0.9, 0.99, 0.99999]  # 5
epsilon_list = [1, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1]  # 8
for epsilon in epsilon_list:
    for beta1 in beta1_list:
        for beta2 in beta2_list:
            set_random_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            dir_outs = '../search_outs/'
            model_out_dir = dir_outs + 'epsilon_' + str(epsilon) + '___beta1_' + str(beta1) + '___beta2_' + str(beta2)
            os.mkdir(model_out_dir)
            # Prepare callbacks for model saving and for learning rate adjustment.
            file_path = model_out_dir + '/' + model_type + '-' + str(dataset_size) + '-{epoch:02d}-{val_loss:.2f}.hdf5'
            checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, mode='min', save_best_only=True)
            csv_logger = CSVLogger(model_out_dir + '/' + 'training.log', separator=',')

            lr_scheduler = LearningRateScheduler(lr_schedule)

            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

            callbacks = [checkpoint, lr_reducer, lr_scheduler]

            # Create compile and train model
            resnet = ResNet(input_shape=input_shape, n=n, output_nodes=1)
            model = resnet.create_model()
            model.summary()
            model.load_weights('weights.h5')
            model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr_schedule(0), epsilon=epsilon, beta_1=beta1, beta_2=beta2), metrics=['mean_squared_logarithmic_error', 'mean_squared_error'])
            history = model.fit_generator(generator=generator_train, steps_per_epoch=num_batches, epochs=epochs, verbose=1, validation_data=generator_validation, callbacks=[csv_logger, checkpoint])

            scores = model.evaluate_generator(generator=generator_test)
            print(scores)


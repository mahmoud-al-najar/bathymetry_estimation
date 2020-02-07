from __future__ import print_function
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import random
from tensorflow import set_random_seed
import os
import csv
import math
from sub_tile_estimation.bathymetry.data.extracts_generator import ExtractsGenerator
from sub_tile_estimation.bathymetry.models.sanity import Sanity
from sub_tile_estimation.bathymetry.models.resnet import ResNet


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
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


seed = 1
set_random_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Model parameters
dataset_size = 25_000
batch_size = 128  # 64
epochs = 200
input_shape = (40, 40, 4)
output_nodes = 1
search_out_dir = ''  # ######################## TO BE SET
input_csv = 'v13.csv'  # ######################## TO BE SET

lr_list = [1e-05]
epsilon_list = [1e-08]
beta1_list = [0.99]
beta2_list = [0.999]
for lr in lr_list:
    for epsilon in epsilon_list:
        for beta1 in beta1_list:
            for beta2 in beta2_list:
                set_random_seed(seed)
                random.seed(seed)
                np.random.seed(seed)

                list_ids = []
                labels = dict()

                # Read data csv
                with open(input_csv, 'r') as f:
                    reader = csv.reader(f)
                    dataset = list(reader)
                random.shuffle(dataset)
                for r in dataset:  # [-dataset_size:]:
                    list_ids.append(r[0])
                    # 1:depth, 2:hmo, 3:freqpeak, 4:thetapeak
                    labels[r[0]] = float(r[1])

                # Setup data generators
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

                generator_train = ExtractsGenerator(partition['train'], labels, x_shape=input_shape,
                                                    y_size=output_nodes, batch_size=batch_size)
                generator_test = ExtractsGenerator(partition['test'], labels, x_shape=input_shape, y_size=output_nodes,
                                                   batch_size=batch_size)
                generator_validation = ExtractsGenerator(partition['validation'], labels, x_shape=input_shape,
                                                         y_size=output_nodes, batch_size=batch_size)

                total_items = len(partition['train'])
                num_batches = int(total_items / batch_size)

                # Create model
                # resnet = ResNet(input_shape=input_shape, output_nodes=output_nodes, n=2)
                # model = resnet.create_model()
                sanity = Sanity(input_shape=input_shape, output_nodes=output_nodes)
                model = sanity.create_model()
                print(model.name)
                model.summary()

                # Setup output directory
                output_dir = search_out_dir + model.name + '_' + str(total_items) + '_lr_' + str(lr) + '__epsilon_' + \
                             str(epsilon) + '__beta1_' + str(beta1) + '__beta2_' + str(beta2) + '/'
                if output_dir != '':
                    if not os.path.isdir(output_dir):
                        os.mkdir(output_dir)

                # Prepare callbacks for model saving and for learning rate adjustment.
                file_path = output_dir + model.name + '_' + str(total_items) + '_{epoch:02d}_{val_loss:.2f}.hdf5'
                checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, mode='min', save_best_only=True)
                csv_logger = CSVLogger(output_dir + 'training.log', separator=',')
                lr_scheduler = LearningRateScheduler(lr_schedule)
                lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
                tb_cb = TensorBoard(log_dir=output_dir, histogram_freq=0, write_graph=True, write_images=True,
                                    update_freq='epoch')
                callbacks = [csv_logger, tb_cb, checkpoint]  # , lr_reducer, lr_scheduler]

                # Compile and train model
                model.compile(loss='mean_squared_error',
                              optimizer=Adam(lr=lr, epsilon=epsilon, beta_1=beta1, beta_2=beta2),
                              metrics=['mean_squared_error'])

                history = model.fit_generator(generator=generator_train, steps_per_epoch=num_batches,
                                              epochs=epochs, verbose=1, validation_data=generator_validation,
                                              callbacks=callbacks)

                scores = model.evaluate_generator(generator=generator_test)
                print(scores)
print('DONE')

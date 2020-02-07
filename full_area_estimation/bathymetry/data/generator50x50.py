import numpy as np
import keras
np.random.seed(448)


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_ids, labels, batch_size=1, x_shape=(200, 200, 4), y_shape=(200, 200, 1), shuffle=True):
        """Initialization"""
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_ids
        self.shuffle = shuffle
        self.on_epoch_end()
        self.indices = None
        self.on_epoch_end()
        self.shape = self.x_shape

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""  # x : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, self.x_shape[0], self.x_shape[1], self.x_shape[2]))
        y = np.empty((self.batch_size, self.y_shape[0], self.y_shape[1], self.y_shape[2]))
        # Generate data
        for i, ID in enumerate(list_ids_temp):
            slice = -1
            if ID.endswith('.npy__0'):
                slice = 0
            elif ID.endswith('.npy__1'):
                slice = 1
            elif ID.endswith('.npy__2'):
                slice = 2
            elif ID.endswith('.npy__3'):
                slice = 3
            ID = ID[:-3]
            burst = np.load(ID)
            burst = burst[:4, :, :]
            burst = np.rollaxis(burst, 0, 3)
            bathy = np.load(self.labels[ID][:-3])
            bathy = np.expand_dims(bathy, axis=2)
            bathy = bathy / 10

            if slice != -1:
                if slice == 0:
                    burst = burst[:50, :50, :]
                    bathy = bathy[:50, :50, :]
                elif slice == 1:
                    burst = burst[50:100, 50:100, :]
                    bathy = bathy[50:100, 50:100, :]
                elif slice == 2:
                    burst = burst[100:150, 100:150, :]
                    bathy = bathy[100:150, 100:150, :]
                elif slice == 3:
                    burst = burst[150:, 150:, :]
                    bathy = bathy[150:, 150:, :]

            x[i, ] = burst
            y[i] = bathy
        return x, y

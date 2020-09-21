# Author :  Jean-Baptiste PROST
# Date : Summer 2020

import os
import warnings
import numpy as np
import pandas as pd
import h5py

from tensorflow.keras.utils import Sequence 
from sklearn.utils import class_weight

from PreProcessing import PreProcessing

### Data Generator Class
class DataGenerator(Sequence):

    def __init__(self, data_directory, ids,
                 batch_size=32,
                 shuffle=True):

        self.data_directory = data_directory  # where to access the samples

        # Get the ids that are intersecting the data base and the partitioning
        ids_in_database = [int(file.split('.')[0]) for file in os.listdir(data_directory)]
        self.list_IDs = list(set(ids) & set(ids_in_database))  # list of samples

        # a parallel indexing for batch access and shuffling
        self.indexes = np.arange(len(ids))

        # if data is shuffle at every epoch, default is True
        self.shuffle = shuffle
        self.batch_size = batch_size  # size of the batch

    def get_IDs(self):
        """
        Returns the id list
        """
        return self.list_IDs
    
    def on_epoch_end(self):
        """
        Over writes the Sequence method : 'Method called at the end of every epoch.'
        """
        raise NotImplementedError

    def _data_generation_(self, list_IDs_temp):
        """
        Access, load and process the data
        """
        raise NotImplementedError

class VideoGenerator(DataGenerator, PreProcessing):
    """
    Inherits form the Sequence class (https://keras.io/utils/)
    Custom data generator that avoid loading the entire dataset into the memory.
    Manages all the data from retrieving to providing to the model.
    Applies transformation at the population scale (uniformization, class proportion),
    and at the sample scale (rotation, shifting, normalization) without any data
    duplication.
    Process batches of data in paralele insInstead, yield batch of data to feed the model.
    
    The class manages binary classification data and regression

    The instance is call by an method of a Keras instance Sequential
    (https://keras.io/models/sequential/)
    or Model (https://keras.io/models/model/).
    The methods are :  fit, evaluate, predict, fit/predict/evaluate _generator.
    """

    def __init__(self, data_directory, ids, labels, dim,
                 pre_processing_dict=None,
                 balanced=0.,
                 testing=False,
                 batch_size=32,
                 shuffle=True,
                 uniform=False):
        """
        INPUTS:
            data_directory = directory of files location [str]
            ids = list of samples [list]
            labels = labels associated with samples [pd.Series]
            dim = dimension of a samples [tuple]
            pre_processing_dict =  transformation to be applied [dic]
            balanced = proportion of positive class [0<float=<0.5 ],
            testing = activate/deactivate testing mode [boolean]
            batch_size = size of a batch [int]
            shuffle = shuffle data at the beginning of epoch [boolean]
            uniform = make the distribution of labels uniform [boolean]
        """
        # - Generator
        DataGenerator.__init__(self, data_directory=data_directory,
                                             ids=ids,
                                             batch_size=batch_size,
                                             shuffle=shuffle)
        self.labels = labels  # corresponding labels

        # - Pre-processing
        PreProcessing.__init__(self, dim= dim,
                               pre_processing=pre_processing_dict)


        # - Edit the distribution of the labels and ids
        self._data_distribution(balanced, uniform)

        if testing:  # testing mode
            self._testing_()  # adjusts the settings to make it a simple iterator

        # Allocate memory once for data loading
        self.npy = '.npy'

        # Display summary of what manipulation have been done
        self._print_info_(testing, uniform)

        # initialize the learning process
        self.on_epoch_end()

    def _print_info_(self, testing, uniform):
        """
        Prints the info/set-up of the Generator when the instance is created
        """
        dir_ = self.data_directory.split('/')[-2]
        # Number of samples
        if len(np.unique(self.labels)) > 3:  # regression
            info = '{} regression samples from {}'.format(len(self.labels), dir_)
            if uniform:
                info = info + '. Uniform distribution (augmented to {} samples)'.format(len(self.list_IDs))

        else:  # binary classification
            if self.balanced:
                info = '{} positive samples and {} negative ones from {}'.format(len(self.list_IDs_pos),
                                                                                 len(self.list_IDs_neg),
                                                                                 dir_)
            elif np.sum(self.labels) :
                info = '{} positive samples and {} negative ones from {}'.format(np.sum(self.labels),
                                                                                 len(self.labels) - np.sum(self.labels),
                                                                                 dir_)
            else:
                info = '{} samples from {}'.format(len(self.labels), dir_)

        if testing:
            print('Testing Mode - ' + info)
        else:
            print(info)

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        Overwritten from Sequence
        """
        if self.balanced:
            # returns the number of samples in the positive class
            return int(np.floor(len(self.list_IDs_pos) / self.batch_size))
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size))

    def get_labels(self):
        """
        Get the labels
        """
        return self.labels

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.balanced:
            self.neg_indexes = np.arange(len(self.list_IDs_neg))
            self.pos_indexes = np.arange(len(self.list_IDs_pos))
            if self.shuffle:
                np.random.shuffle(self.pos_indexes)
                np.random.shuffle(self.neg_indexes)

        else:
            if self.shuffle:
                np.random.shuffle(self.indexes)
            else:
                pass

    def get_class_weights(self):
        """
        Returns the proportion of classes
        """
        if len(np.unique(self.labels)) == 2:
            if self.balanced:
                return {1: len(self.neg_indexes) / len(self.pos_indexes),
                        0: len(self.pos_indexes) / len(self.neg_indexes)}
            else:
                class_w = class_weight.compute_class_weight('balanced',
                                                            np.unique(self.labels),
                                                            self.labels)
                return {0: class_w[0], 1: class_w[1]}
        else:
            raise ValueError('Labels are not for binary classification. Class weight is implemented for binary '
                             'classification only.')

    # --- Data Distribution ---

    def _data_distribution(self, balanced, uniform):
        """
        Varying proportion of class proportions.
        If the task is a binary classification, it changes the balanced of the positive class if over-sampling needs to
        be performed.
        If the task is a regression, the data can be uniformized according to the labels.
        """
        if len(np.unique(self.labels)) > 3:  # if regression task
            balanced = False  # ensures that the data will not be balanced

        if (balanced < 0) or (balanced > 0.5):
            raise ValueError('The proportion argument must be between 0 and 0.5')
        else:
            if balanced:
                self.balanced = True  # transform to binary argument
                self._balance_data_(balanced)  # transform proportion
                self.neg_indexes = []
                self.pos_indexes = []
            else:
                self.balanced = False
                self.indexes = np.arange(len(self.list_IDs))

        if uniform:  # makes the distribution of labels uniform
            self.list_IDs = self._uniform_()

    def _uniform_(self):
        """
        Makes the distribution unifrom of labels value.
        Used for regression only.
        """
        # count the max occurrence of a label
        counts = self.labels.value_counts()
        max_counts = counts.max()
        df_uniform = pd.Series()

        # for each count associated with each label
        for value, count in zip(counts.index, counts):
            sub_arr = self.labels[self.labels == value]
            ratio_sub = max_counts / len(sub_arr)

            # more than Twice, add the entire sub-set
            if ratio_sub > 2:
                for n in range(int(ratio_sub) - 1):
                    sub_arr = sub_arr.append(self.labels[self.labels == value])

            # rest of the division, add until reaching the max_count
            diff_sub = max_counts - len(sub_arr)
            if diff_sub != 0:
                sub_arr = sub_arr.append(self.labels[self.labels == value].iloc[:diff_sub])
            df_uniform = pd.concat([df_uniform, sub_arr])

        return list(df_uniform.index)

    def _balance_data_(self, balanced):
        """
        Duplicates positive IDs such that positive samples will be augmented
        """
        # list of natural numbers for indexing
        self.list_IDs_neg = list(self.labels[self.labels == 0].index)
        self.list_IDs_pos = self.labels[self.labels == 1].index
        nb_samples = len(self.list_IDs_neg)

        if len(self.list_IDs_pos) < int(balanced * nb_samples):
            # tiles the positive index to reach the same number of positive as the negative
            tile_nb = int((balanced * nb_samples) / len(self.list_IDs_pos))

            if tile_nb < 1:
                pass
            else:
                self.list_IDs_pos = list(np.tile(self.list_IDs_pos, tile_nb))
            # tile_nb can only be int, ad the remaining fraction to reach the desired length
            self.list_IDs_pos = self.list_IDs_pos + self.list_IDs_pos[
                                                    :int((balanced * nb_samples) - len(self.list_IDs_pos))]
        else:
            self.list_IDs_pos = self.list_IDs_pos[:int(balanced * nb_samples)]

    def _testing_(self):
        """
        For testing mode, ensures that the following parameters are correctly set
        """
        self.shuffle = False
        self.batch_size = 1
        self.uniform = False
        self.normalize = False
        self.shift = False
        self.rotate = False

    # --- Data reaching ---
    def _data_generation_(self, list_IDs_temp):
        """
        Generates data containing batch_size samples.
        Samples must be structured as [nb of frames, height, width]
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Load sample
            try:
                x = np.load(self.data_directory + str(ID) + self.npy)
            except FileNotFoundError:
                warnings.warn( str(ID)+ self.npy +' not found')
            
            # live data augmentation
            X[i,] = self._preprocess_sample_(x)
            # store label
            y[i] = self.labels[ID]

        return X, y

    def __getitem__(self, index):
        """
        Is called during by 'fit, evaluate, predict, fit/predict/evaluate _generator'.
        Returns a batch of pre-processed samples
        Overwrittes a methode in Sequence
        """
        if self.balanced:
            # Generate indexes of the batch
            pos_idx = self.pos_indexes[int(index * self.batch_size / 2):int((index + 1) * self.batch_size / 2)]
            neg_idx = self.neg_indexes[int(index * self.batch_size / 2):int((index + 1) * self.batch_size / 2)]
            # Find list of IDs
            list_IDs_temp = [self.list_IDs_pos[k] for k in pos_idx]
            list_IDs_temp = list_IDs_temp + [self.list_IDs_neg[k] for k in neg_idx]

        else:
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
            
        return self._data_generation_(list_IDs_temp)      
    
class TestVideoGenerator(VideoGenerator):
    """
    Creates a DataGenerator in testing mode with explicit name
    """
    def __init__(self, data_directory, ids, labels, dim):
        super().__init__(data_directory, ids, labels, dim,
                        batch_size=1,
                        tesing=True)

        
class AutoEncoderGenerator(DataGenerator, PreProcessing):

    def __init__(self, data_directory, ids, dim, pre_processing_dict=None,
                 batch_size=32,
                 shuffle=True):
        """
        INPUTS:
            data_directory = directory of files location [str]
            ids = list of samples [list]
            dim = dimension of a samples [tuple]
            pre_processing_dict =  transformation to be applied [dic]
            balanced = proportion of positive class [0<float=<0.5 ],
            testing = activate/deactivate testing mode [boolean]
            batch_size = size of a batch [int]
            shuffle = shuffle data at the beginning of epoch [boolean]
            uniform = make the distribution of labels uniform [boolean]
        """
        # - Generator
        DataGenerator.__init__(self, data_directory=data_directory,
                                             ids=ids,
                                             batch_size=batch_size,
                                             shuffle=shuffle)
        # - Pre-processing
        PreProcessing.__init__(self, dim=dim,
                               pre_processing=pre_processing_dict)


        # allocates the memory once for all
        self.npy = '.npy'

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)
        else:
            pass

    def _data_generation_(self, list_IDs_temp):
        """
        Generates data containing batch_size samples.
        Samples must be structured as [nb of frames, height, width]
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Load sample
            try:
                x = np.load(self.data_directory + str(ID) + self.npy)
                try:
                    # pre-processing
                    X[i,] = self._preprocess_sample_(x)

                except (ValueError, KeyError):
                    warnings.warn('Shape or key-error {} sample'.format(ID))

            except FileNotFoundError:
                warnings.warn(ID, ' not found')

        return X, X

    def __getitem__(self, index):
        """
        Generate a batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        return self._data_generation_(list_IDs_temp)
    
class ArrayGenerator(Sequence):
    """
    Loads batches of arrays into the memory.
    The data X must be provided with its corresponding labels Y. X can be supplemented with additional information
    called 'co_factors' that would be feed into the neural network in different layers.


    The instance is call by an method of a Keras instance Sequential
    (https://keras.io/models/sequential/)
    or Model (https://keras.io/models/model/).
    The methods are :  fit, evaluate, predict, fit/predict/evaluate _generator.
    """
    def __init__(self, x_set, y_set,
                 co_factors=None,
                 batch_size =32):
        """
         x_set: [str] directory of a h5 file OR [np.array]
         y_set: [pandas.Series] labels
         co_factors: [tuple([np.array](int or float))]
         batch_size: [int]
        """

        self.batch_size = batch_size

        # - Data access
        if not isinstance(x_set, str): # X of type array
            self.x = x_set
            self.load_all = True
        elif isinstance(x_set, str) and x_set.endswith('.h5'): # X is the directory to a .h5 file
            self.x = x_set  # is a path to a hdf5 file
            self.load_all = False
        else:
            raise TypeError('The data is neither a array nor a H5 file directory')

        # Co factor data is available
        if co_factors is not None:
            y_set = y_set.to_frame()
            y_set['id_nb'] = range(len(y_set)) # create new index
            self.tabular = y_set

            del y_set # free some memory explicitly

            # concatenates the co-factors into a tabular form
            for co_factor in co_factors:
                self.tabular = pd.concat((self.tabular, co_factor), axis=1)
                self.tabular = self.tabular.dropna(how='any')
            self.tabular.rename(columns={self.tabular.columns[0]: 'labels'}, inplace=True)

            if self.load_all:
                # select the data that has co-factors
                self.x = x_set[self.tabular.id_nb.astype(int).to_list()]
            else:
                # get the ids that have co-factors
                self.x_idx = self.tabular.id_nb.astype(int).to_list()

            # tabular is the co-factor array
            self.tabular.drop(columns='id_nb', inplace=True)
            self.labels = self.tabular.labels

        else: # no co-factors
            self.labels = y_set
            self.__getitem__ = self._getitem_array

    def __len__(self):
        """
        Returns the number of batches to process
        """
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def get_class_weights(self):
        """
        Returns the proportion of classes
        """
        class_w = class_weight.compute_class_weight('balanced',
                                                    np.unique(self.labels),
                                                    self.labels)
        return {0: class_w[0], 1: class_w[1]}

    def _getitem_array(self, idx):
        """
        Returns batches of data and labels.
        """
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

    def __getitem__(self, idx):
        """
        Returns batches of data, co-factors and labels.
        Is called during by 'fit, evaluate, predict, fit/predict/evaluate _generator'.
        Returns a batch of pre-processed samples
        Over-writtes a method in Sequence
        """

        if self.load_all:
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            # load data by partition
            idx_tmp = self.x_idx[idx * self.batch_size:(idx + 1) * self.batch_size]

            with h5py.File(self.x, 'r') as f:
                batch_x = f['X_train'][idx_tmp]

        batch_y = self.tabular.labels.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_tab = self.tabular.iloc[idx * self.batch_size:(idx + 1) * self.batch_size, 1:]

        return [np.array(batch_x), np.array(batch_tab)], np.array(batch_y)
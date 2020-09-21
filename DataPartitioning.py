# Author :  Jean-Baptiste PROST
# Date : Summer 2020

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data_fold(k_fold, csv_dir, target):
    """
    Get the data previously split in k-folds for cross validation in .csv files
    containing the patient ids ('eid'), their age and their sex.
    Each fold is decomposed into 3 set : one for each training, validation and testing.

    Inputs:
        - k_fold [int] :  the k-th fold to consider
        - csv_dir [str] : directory of the csv file. Must be of form: "..._{}_{}.csv" so that
         the corresponding train/test/val and k-fold can be extracted.

    Outputs:
        - partition [dict] :  dict for train/test/val containing a list of
        int patient ids
        - labels [dict] : dict for train/test/val containing a pd.Series with patient
        (int) ids as index and the corresponding value of interest (Age or Sex)
    """

    folds = ['train', 'test', 'val']
    partition = dict()
    labels = dict()

    for f in folds:
        data_path = csv_dir.format(f, k_fold)
        data = pd.read_csv(data_path, index_col='eid')
        if isinstance(data.index[0], str):
            data.index = data.index.map(lambda x: int(x[:-4]))
        else:
            pass
        partition[f] = data.index.to_list()
        labels[f] = data[target]

    return partition, labels


class DataPartitioning():
    """
    This class splits the data into train, validation and testing sets.
    """

    def __init__(self, labels, test_fraction=0.1, val_fraction=0.1, **kwargs):
        """
        labels: pandas.Series labels indexed by sample id
        test_fraction: fraction of the data-set for the testing set
        val_fraction: fraction of the data-set for the validation set
        kwargs: enables to define the list of ids>
            _ data_directory : reads the files present in the folder
            _ id_list : list of ids encoded as integers
        """
        self.test_frac = test_fraction
        self.val_frac = val_fraction

        if 'data_directory' in kwargs:
            directory = kwargs['data_directory']
            # get the list of files in a directory and removes the format extension to convert to int
            self.ids = [int(file.split('.')[0]) for file in os.listdir(directory)]
        elif 'id_list' in kwargs:
            self.ids = kwargs['id_list']
        else:
            raise ValueError('Please provide either a `data_directory` string or an `id_list` of ints')

        if len(labels.unique()) == 1:
            df_labels = pd.DataFrame(index=self.ids,
                                     columns=['Labels'],
                                     data=np.zeros(len(self.ids)).astype(int))

            # get rid of absent samples
            positive = set(self.labels.index) - (set(self.labels.index) - set(ids))
            df_labels.loc[positive] = 1
            self.labels = df_labels
        else:
            self.labels = labels

        if 'seed' in kwargs:
            self.random_state = kwargs['seed']
        else:
            self.random_state = None

    def make_partition(self):
        """
        Returns :
        _ partition : dict of list of integers ids
        _ labels : dict of pandas.Series of labels
        """

        X_train, X_test, y_train, y_test = train_test_split(self.ids, self.labels,
                                                            test_size=(self.test_frac + self.val_frac),
                                                            random_state=self.random_state)

        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                        test_size=(self.val_frac / (self.test_frac + self.val_frac)),
                                                        random_state=self.random_state)

        partition = {'train': X_train, 'test': X_test, 'val': X_val}
        labels = {'train': y_train, 'test': y_test, 'val': y_val}

        return partition, labels


# Author :  Jean-Baptiste PROST
# Date : Summer 2020

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from pandas.errors import EmptyDataError
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, f1_score, log_loss, average_precision_score

from DataGenerators import VideoGenerator

class CustomCallback(tf.keras.callbacks.Callback):
    """
    This class enabales to gather multiple callbacks into a single one.
    It has several roles :
    1) Evaluate the validation set throught training at the end of each epoch.
       This is could be computed in the `model.fit` function. However, it is done
       batch wise and averages the batch scores at the end of the epoch.
       For R2 metric, this could cause some issues. In this class, the scores are 
       ccomputed with allt eh data at the end of the epoch.
    2) Save models through training (model check-point)
    3) Keep track of the performance and the loss
    4) Stop training when learning has stopped (early-stopping)
    4) Terminate training if nan occurs (terminate on NaN)

    This class would be a `base class`> The final version should contain custom loss 
    and metric functions.
    """

    def __init__(self, validation_data, patience, restore_best_weights, baseline=False, output_dir=False,
                 input_dir=False, restore=True, save_best=False, save=True):

        super().__init__()

        # Adapts to different type of validation data
        if isinstance(validation_data, VideoGenerator) :
            self.val_data = validation_data
            self.val_labels = validation_data.get_labels()
            self.val_data.batch_size = 1  # make sure the batch_size is 1
            self.get_predictions = self.get_predictions_generator

        elif isinstance(validation_data, tuple):
            self.val_data = validation_data[0]
            self.val_labels = validation_data[1]
            self.get_predictions = self.get_predictions_array

        else:
            raise TypeError('Validation Data is not in the correct format. VideoGenerator or tuple (X,y)')

        self.monitor_op = np.greater  # operation to compare the evolution of the validation score
        self.baseline = baseline  # baseline score for training resuming
        self.patience = patience  # how many epoch before launching early-stopping
        self.wait = 0 # how many epoch with no improvement before early-stopping
        self.stopped_epoch = 0 # at which epoch it stops

        self.save_best = save_best # whether to save only the best score or all epochs
        self.best_val_score = -1e6 # very low value 
        # whether to restore the best model according to the validation score
        self.restore_best_weights = restore_best_weights
        # initialize the best weights of the NN
        self.best_weights = None
        # whether to save the weights of the model
        self.save = save

        # directory for saving and reading the weights and training files
        if output_dir:
            self.output_dir = output_dir + '/'
        else:
            self.output_dir = os.getcwd() + '/' # current working directory

        if input_dir:
            self.input_dir = input_dir + '/'
        else:
            self.input_dir =  os.getcwd() + '/'

        # resume training if score.csv is found in the directory
        if ('score.csv' in os.listdir(self.input_dir)) and restore:
            print('Callback : Load existing Scores and Losses.')
            self.train_loss = list(np.load(self.input_dir + 'Tr_loss.npy'))

            # error if the score.csv file exist but is empty
            try:
                self.df_score = pd.read_csv(self.input_dir + 'score.csv')
                self.idx_epoch = self.df_score.shape[0]
            except EmptyDataError:
                print('Empty CSV')
                self.idx_epoch = 0
                self.df_score = pd.DataFrame(index=range(self.idx_epoch))
        else:
            print('Callback : No score file found, first training.')
            self.df_score = pd.DataFrame()
            self.train_loss = []
            self.idx_epoch = 1

    def on_train_begin(self, logs=None):
        """
        Set the parameters for a new training
        """
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def terminateOnNan(self, batch, logs):
        """
        Stops the learning process if a Nan is produced
        """
        logs = logs or {}
        loss = logs.get('loss')

        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True

    def earlyStopping(self, epoch, logs, score):
        """
        Stops the training process if the validation score does not improve
        """

        if self.monitor_op(score, self.best):
            self.best = score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()

        else:
            self.wait += 1
            if self.wait >= self.patience:
                print('Validation R2 not imporving, stop training')
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_batch_end(self, batch, logs=None):

        self.train_loss.append(logs.get('loss'))
        self.terminateOnNan(batch, logs)

    def get_predictions_generator(self):
        """
        Computes the predictions of the validation DataGenerator
        """
        return self.model.predict_generator(self.val_data)

    def get_predictions_array(self):
        """
        Computes the predictions of the validation data array
        """
        return self.model.predict(self.val_data)

    def save_model(self, validation_score):
        """
        Saves the model at each epoch or only the best model
        """
        if self.save_best:
            if self.best_val_score < validation_score:
                self.best_val_score = validation_score

                str_ = self.output_dir + 'epoch_{:04.0f}.h5'.format(self.idx_epoch)
                self.model.save(str_)
                print('Best Model epoch_{:04.03f}.h5 saved - {}'.format(self.idx_epoch,
                                                                        np.round(self.best_val_score, 3)))

        else:
            str_ = self.output_dir + 'epoch_{:04.0f}.h5'.format(self.idx_epoch)
            self.model.save(str_)
            print('Model saved at ' + str_)

    def on_epoch_end(self, epoch, logs=None):
        raise NotImplementedError("Subclasses should implement this!")

class R2Callback(CustomCallback):
    """
    Uses the R-squared metric as monitoring, and RMSE as loss function.
    """

    def __init__(self, validation_data, patience, restore_best_weights, baseline=None,
                 output_dir='./', input_dir=False, restore=True, save_best=False,
                 save=True):
        super().__init__(validation_data, patience,
                         restore_best_weights, baseline,
                         output_dir, input_dir, restore, save_best, save)

    def on_epoch_end(self, epoch, logs=None):
        """

        """
        # get the predictions of the current model
        preds = self.get_predictions()

        # compute the validation scores
        validation_score = r2_score(self.val_labels, preds)
        RMSE_tmp = np.sqrt(mean_squared_error(self.val_labels, preds))

        print('R2 val : {} , loss val : {}'.format(np.round(validation_score, 3), np.round(RMSE_tmp, 3)))

        # save the training and validation scores
        # save to the log, to be accessible by History callback
        logs['R2_val'] = validation_score
        logs['loss_val'] = RMSE_tmp
        
        self.df_score.loc[self.idx_epoch, 'R2_val'] = validation_score
        self.df_score.loc[self.idx_epoch, 'RMSE_val'] = RMSE_tmp
        self.df_score.loc[self.idx_epoch, 'RMSE_tr'] = logs.get('loss')
        self.df_score.loc[self.idx_epoch, 'R2_tr'] = logs.get('R2_')

        # Savings
        if self.save:
            self.df_score.to_csv(self.output_dir + 'score.csv', index=False)
            np.save(self.output_dir + 'Tr_loss.npy', np.array(self.train_loss))
            self.save_model(validation_score)

        # whether to stop the training process
        self.earlyStopping(epoch, logs, validation_score)
        self.terminateOnNan(epoch, logs)

        # epoch counter
        self.idx_epoch += 1

class F1Callback(CustomCallback):
    """
    Uses the F-1 metric as monitoring, and binary cross entropy as loss function.
    The average-precision-recall and the f1 score are computed.
    A custom early-stopping function terminates the training processes if the F1
    score is null.
    """

    def __init__(self, validation_data, patience, restore_best_weights, baseline=None,
                 output_dir=None, input_dir=None, restore=True, save_best=False,
                 save=True):

        super().__init__(validation_data, patience,
                         restore_best_weights, baseline,
                         output_dir, input_dir, restore, save_best, save)

        self.null_f1 = 0

    def terminateNullF1(self, F1):
        """ Ends training if F1 is zero"""

        if F1 < 1e-5:
            self.null_f1 += 1

            if self.null_f1 == 10:
                print('Null F1. Stop Training')
                self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        """
        Computes the metrics and loss for the validation and save
        the model. 
        """
        # gets the predictions
        preds = self.get_predictions()

        # compute metrics
        loss_tmp = log_loss(self.val_labels, preds)
        avg_prec = average_precision_score(self.val_labels, preds, pos_label=1)

        # copy for dirscretization at 0.5 threshold
        discr = preds.copy()
        discr[discr > 0.5] = 1
        discr[discr <= 0.5] = 0
        f1_tmp = f1_score(self.val_labels, discr)

        print('f1 val : {} , loss val : {}, Avg. Prec. : {}'.format(np.round(f1_tmp, 2), np.round(loss_tmp, 2),
                                                                    np.round(avg_prec, 2)))

        # save scores
        # save to the log, to be accessible by History callback
        logs['AvgPrec_val'] = avg_prec
        logs['f1_val'] = f1_tmp
        logs['loss_val'] = loss_tmp
        
        self.df_score.loc[self.idx_epoch, 'AvgPrec_val'] = avg_prec
        self.df_score.loc[self.idx_epoch, 'f1_val'] = f1_tmp
        self.df_score.loc[self.idx_epoch, 'loss_val'] = loss_tmp
        self.df_score.loc[self.idx_epoch, 'loss_tr'] = logs.get('loss')
        self.df_score.loc[self.idx_epoch, 'f1_tr'] = logs.get('f1_')
        self.df_score.to_csv(self.output_dir + 'score.csv', index=False)


        # Savings
        if self.save:
            np.save(self.output_dir + 'Tr_loss.npy', np.array(self.train_loss))
            self.df_score.to_csv(self.output_dir + 'score.csv')
            self.save_model(f1_tmp)

        self.earlyStopping(epoch, logs, f1_tmp)
        self.terminateNullF1(f1_tmp)
        self.terminateOnNan(epoch, logs)

        #epoch counter
        self.idx_epoch += 1
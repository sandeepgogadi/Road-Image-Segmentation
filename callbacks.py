#!/usr/bin/python3

import tensorflow as tf
import numpy as np

# To change lr to learning_rate for tf.train compatibility


class LearningRateScheduler(tf.keras.Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        learning_rate = float(K.get_value(self.model.optimizer.learning_rate))
        try:  # new API
            learning_rate = self.schedule(epoch, learning_rate)
        except TypeError:  # old API for backward compatibility
            learning_rate = self.schedule(epoch)
        if not isinstance(learning_rate, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.learning_rate, learning_rate)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, learning_rate))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['learning_rate'] = K.get_value(self.model.optimizer.learning_rate)


class PolyDecay:
    def __init__(self, initial_lr, power, n_epochs):
        self.initial_lr = initial_lr
        self.power = power
        self.n_epochs = n_epochs

    def scheduler(self, epoch):
        return self.initial_lr * np.power(1.0 - 1.0*epoch/self.n_epochs, self.power)


def get_callbacks(args):

    checkpoint = tf.keras.callbacks.ModelCheckpoint('weights/weights_{}.h5'.format(args.net),
                                                    monitor='val_conv6_cls_categorical_accuracy', mode='max', save_weights_only=True,
                                                    save_best_only=True, verbose=1)

    stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_conv6_cls_categorical_accuracy', patience=10)

    lr_decay = LearningRateScheduler(PolyDecay(args.lr, 0.9, 10).scheduler)

    terminate = tf.keras.callbacks.TerminateOnNaN()

    return [checkpoint, stopping, lr_decay, terminate]

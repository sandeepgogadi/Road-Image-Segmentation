#!/usr/bin/python3

import tensorflow as tf
import numpy as np


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

    lr_decay = tf.keras.callbacks.LearningRateScheduler(PolyDecay(args.lr, 0.9, 10).scheduler)

    terminate = tf.keras.callbacks.TerminateOnNaN()

    return [checkpoint, stopping, lr_decay, terminate]

#!/usr/bin/python3

import tensorflow as tf


def get_callbacks(args):

    checkpoint = tf.keras.callbacks.ModelCheckpoint('weights/weights_{}.h5'.format(args.net),
                                                    monitor='val_conv6_cls_categorical_accuracy', mode='max', save_weights_only=True,
                                                    save_best_only=True, verbose=1)

    stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_conv6_cls_categorical_accuracy', patience=10)

    lr_decay = tf.keras.callbacks.LearningRateScheduler(PolyDecay(args.lr, 0.9, 10).scheduler)

    terminate = tf.keras.callbacks.TerminateOnNaN()

    return [checkpoint, stopping, lr_decay, terminate]

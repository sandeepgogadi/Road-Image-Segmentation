#!/usr/bin/python3

import argparse
import models
import os
from keras import backend as K
import tensorflow as tf
from data_generator import *
from history import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size.')
parser.add_argument('--epochs', type=int, default=10,
                    help='total epochs.')
parser.add_argument('--width', type=int, default=640,
                    help='image width.')
parser.add_argument('--height', type=int, default=320,
                    help='image height.')
parser.add_argument('--num_classes', type=int, default=20,
                    help='number of classes.')
parser.add_argument('--weights_path', type=str, default=None,
                    help='Specify weights path.')
parser.add_argument('--train', type=bool, default=True,
                    help='train flag')
parser.add_argument('--net', type=str,
                    help='Choice of FCN8, ICNET, PSPNET, SEGNET, UNET')
parser.add_argument('--use_tpu', type=bool, default=False,
                    help='bool for TPU use.')
parser.add_argument('--custom_data', type=bool, default=False,
                    help='Specify if you want to train with custom data')
parser.add_argument('--lr', type=float, default=.0001,
                    help='learning rate for optimizer')
# parser.add_argument('', type=int, default=, help='')

args = parser.parse_args()
print(args)

# Workaround to forbid tensorflow from crashing the gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# Callbacks
callbacks = None

# Data Generators
train_generator = DataGenerator(batch_size=args.batch_size,
                                resize_shape=(args.width, args.height))
val_generator = DataGenerator(mode='val', batch_size=args.batch_size,
                              resize_shape=(args.width, args.height))

# Class weights
class_weights = get_class_weights(args)

# Loss weights
if args.net == 'ICNET':
    loss_weights = [1.0, 0.4, 0.16]
else:
    loss_weights = None

# Optimizer
optimizer = tf.keras.optimizers.Adam(lr=lr)

# Model
model = models.get_model(args)

# TPU
if args.use_tpu:
    TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    strategy = tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER))
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy)

# Model compile
model.compile(optim, 'categorical_crossentropy',
              loss_weights=loss_weights, metrics=['categorical_accuracy'])

# Training
history = model.fit_generator(train_generator, len(train_generator),
                              args.epochs, callbacks=callbacks,
                              validation_data=val_generator,
                              class_weight=class_weights,
                              validation_steps=len(val_generator),
                              workers=2, shuffle=True)

# Plot as save history
plot_history(history)
save_history(history)

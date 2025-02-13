#!/usr/bin/python3

import argparse
import models
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from data_generator import *
from history import *
from utils import prepare_repo
from callbacks import get_callbacks

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
                    help='Choice of ICNET as of now, more networks to be added sooner.')
parser.add_argument('--use_tpu', type=bool, default=False,
                    help='bool for TPU use.')
parser.add_argument('--custom_data', type=bool, default=False,
                    help='Specify if you want to train with custom data')
parser.add_argument('--lr', type=float, default=.01,
                    help='learning rate for optimizer')
# parser.add_argument('', type=int, default=, help='')

args = parser.parse_args()
print(args)

# Clear Session
tf.keras.backend.clear_session()

# Prepare Repo
prepare_repo()

# Workaround to forbid tensorflow from crashing the gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# Callbacks
callbacks = get_callbacks(args)
print('Assigned callbacks!')

# Data Generators
train_generator = DataGenerator(args.num_classes, args.width, args.height, args)
val_generator = DataGenerator(args.num_classes, args.width, args.height, args, mode='val')
print('Alloted generators')

# Class weights
if args.net == 'ICNET':
    class_weights = get_class_weights(args)
else:
    class_weights = None
print('Loaded class weights!')

# Loss weights
if args.net == 'ICNET':
    loss_weights = [1.0, 0.4, 0.16]
else:
    loss_weights = None
print('loaded loss weights')

# Model
model = models.get_model(args.net, args.width, args.height,
                         args.num_classes, args.weights_path, args.train)
print('loaded model')

# TPU
if args.use_tpu:
    TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)
    strategy = tf.contrib.tpu.TPUDistributionStrategy(resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy)
    session_master = resolver.master()
    print('TPU setup completed')
else:
    session_master = ''

# Optimizer
optimizer = tf.keras.optimizers.Adam(args.lr)
print('Optimizer selected')

# Model compile
model.compile(optimizer, 'categorical_crossentropy',
              loss_weights=loss_weights, metrics=['categorical_accuracy'])
print('Model compiled')

# Training
print('Training begin')
history = model.fit_generator(train_generator, len(train_generator),
                              args.epochs, callbacks=callbacks,
                              validation_data=val_generator,
                              class_weight=class_weights,
                              validation_steps=len(val_generator),
                              workers=2, shuffle=True)
print('Training end')

# Plot as save history
plot_history(args, history)
save_history(args, history)

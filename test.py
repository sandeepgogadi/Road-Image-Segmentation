#!/usr/bin/python3

import argparse
import models
import os
import time
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from data_generator import *
from utils import prepare_repo
from label import labels
from matplotlib import pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, help='From a selection of ICNET.')
parser.add_argument('--use_tpu', type=bool, default=False,
                    help='bool for TPU use.')
parser.add_argument('--batch_size', type=int, default=64, help='Specify batch size')
parser.add_argument('--custom_data', type=bool, default=False,
                    help='Specify if you want to train with custom data')
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

# Model params
if args.net == 'ICNET':
    width = 640
    height = 320
    num_classes = 20
    weights_path = 'weights/weights_{}.h5'.format(args.net)

# Data Generators
test_generator = DataGenerator(num_classes, width, height, args, mode='test')
print('Alloted generators')

# Model
model = models.get_model(args.net, width, height, num_classes, weights_path)
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
optimizer = tf.keras.optimizers.Adam(0.0001)
print('Optimizer selected')

# Model compile
model.compile(optimizer, 'categorical_crossentropy', metrics=['categorical_accuracy'])
print('Model compiled')

# Testing
print('Testing begin')
evaluate = model.evaluate_generator(test_generator, steps=len(test_generator))
print('Testing end')

print(evaluate)

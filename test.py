#!/usr/bin/python3

import argparse
import models
import os
import time
import cv2
from keras import backend as K
import tensorflow as tf
from data_generator import *
from utils import prepare_repo
from label import labels
from matplotlib import pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, help='From a selection of ICNET.')
parser.add_argument('--file', type=str, help='File path for the image.')

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

# Model
model = models.get_model(args.net, width, height, num_classes, weights_path)
print('loaded model')

# Load and resize image
img = cv2.imread(args.file, 1)
img = cv2.resize(img, (width, height))
img_reshape = img.reshape(-1, *img.shape)

# Predict
start_time = time.time()
pred = model.predict(img_reshape, batch_size=1)
duration = time.time() - start_time
print('Generated segmentations in %s seconds -- %s FPS' % (duration, 1.0/duration))

# Classes in image
pred_img = np.argmax(pred[0], axis=-1)
print(pred_img.shape)
unique = np.unique(pred_img)
print(unique)

# Prepare color image
col_img = np.zeros((*pred_img.shape, 3), dtype=np.uint8)

for label in labels:
    color = label.color
    trainId = label.trainId
    col_img[pred_img == trainId] = color

# plt.imshow(col_img)
# plt.axis('off')
# plt.show()

# Show images
rgb = img[..., ::-1]
rgb = cv2.resize(rgb, (col_img.shape[1], col_img.shape[0]))

# plt.figure(figsize=(9, 14))
plt.figure()

plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(col_img)
plt.axis('off')

plt.show()

# Save images
cv2.imwrite('outputs/test.png', img)
plt.imsave('outputs/rgb.png', rgb)
plt.imsave('outputs/color.png', col_img)

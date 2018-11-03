#!/usr/bin/python3

import argparse
import models
import os
import time
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from utils import prepare_repo
from label import labels
from matplotlib import pyplot as plt
import numpy as np
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='ICNET', help='From a selection of ICNET.')
parser.add_argument('--file', type=str, help='File path for the image.')
parser.add_argument('--folder', type=str, default='Images/', help='Folder path for the image.')

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

# Images list
if args.file:
    images_list = [args.file]
else:
    images_list = glob.glob(args.folder + '*')

num_images = len(images_list)
print('{} images found.'.format(num_images))

# Predict
start_time = time.time()

# plt.figure(figsize=(9, 14))
plt.figure()

for count, image_file in enumerate(images_list):

    # Load and resize image
    img = cv2.imread(image_file, 1)
    img = cv2.resize(img, (width, height))
    img_reshape = img.reshape(-1, *img.shape)

    pred = model.predict(img_reshape, batch_size=1)

    # Classes in image
    pred_img = np.argmax(pred[0], axis=-1)
    unique = np.unique(pred_img)

    # Prepare color image
    col_img = np.zeros((*pred_img.shape, 3), dtype=np.uint8)

    for label in labels:
        color = label.color
        trainId = label.trainId
        col_img[pred_img == trainId] = color

    # Show images
    rgb = img[..., ::-1]
    rgb = cv2.resize(rgb, (col_img.shape[1], col_img.shape[0]))

    plt.subplot(num_images, 2, 2*count+1)
    plt.imshow(rgb)
    plt.axis('off')

    plt.subplot(num_images, 2, 2*count+2)
    plt.imshow(col_img)
    plt.axis('off')

duration = time.time() - start_time
print('Generated segmentations for %s images in %s seconds -- %s FPS' %
      (num_images, duration, num_images/duration))

plt.show()

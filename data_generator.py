import os
import pickle
import numpy as np
import glob
import cv2
import random
import gc
from tensorflow.keras.utils import to_categorical, Sequence


def get_class_weights(args):

    print('Getting class weights')

    num_classes = args.num_classes
    class_weights_file = "class_weights_{}_{}.p".format(args.width, args.height)

    if os.path.isfile(class_weights_file):
        class_weights = pickle.load(open(class_weights_file, "rb"))
    else:
        class_weights_arr = np.zeros(num_classes)
        labels_list = sorted(glob.glob('data/train/labels/*'))

        for idx in range(len(labels_list)):
            img_weights = np.zeros(num_classes)
            img = cv2.imread(labels_list[idx], 0)
            img = cv2.resize(img, (args.width, args.height))
            for i in range(num_classes):
                if i == 19:
                    img_weights[i] = (img == 255).sum()
                else:
                    img_weights[i] = (img == i).sum()
            img_weights = img_weights/(img.shape[0]*img.shape[1])
            class_weights_arr += img_weights

        class_weights_arr /= len(labels_list)

        class_weights = {}

        for i in range(num_classes):
            class_weights[i] = class_weights_arr[i]

        pickle.dump(class_weights, open(class_weights_file, "wb"))

    return class_weights


class DataGenerator(Sequence):
    def __init__(self, num_classes, width, height, args, folder='data', mode='train',
                 resize_shape=None, brightness=0.1):

        self.image_path_list = sorted(glob.glob(os.path.join(folder, mode, 'images/*')))
        self.label_path_list = sorted(glob.glob(os.path.join(folder, mode, 'labels/*')))
        self.mode = mode
        self.n_classes = num_classes
        self.resize_shape = (width, height)
        self.brightness = brightness
        self.net = args.net
        self.batch_size = args.batch_size
        self.custom_data = args.custom_data

        # Preallocate memory
        if self.net == 'ICNET':
            self.X = np.zeros(
                (self.batch_size, self.resize_shape[1], self.resize_shape[0], 3), dtype='float32')
            self.Y1 = np.zeros(
                (self.batch_size, self.resize_shape[1]//4, self.resize_shape[0]//4, self.n_classes), dtype='float32')
            self.Y2 = np.zeros(
                (self.batch_size, self.resize_shape[1]//8, self.resize_shape[0]//8, self.n_classes), dtype='float32')
            self.Y3 = np.zeros(
                (self.batch_size, self.resize_shape[1]//16, self.resize_shape[0]//16, self.n_classes), dtype='float32')
        else:
            self.X = np.zeros(
                (self.batch_size, self.resize_shape[1], self.resize_shape[0], 3), dtype='float32')
            self.Y = np.zeros(
                (self.batch_size, self.resize_shape[1], self.resize_shape[0], self.n_classes), dtype='float32')

    def __len__(self):
        return len(self.image_path_list) // self.batch_size

    def __getitem__(self, i):
        for n, (image_path, label_path) in enumerate(zip(self.image_path_list[i*self.batch_size:(i+1)*self.batch_size],
                                                         self.label_path_list[i*self.batch_size:(i+1)*self.batch_size])):

            image = cv2.imread(image_path, 1)
            label = cv2.imread(label_path, 0)
            if not self.custom_data:
                label[label == 255] = 19
            if self.resize_shape:
                image = cv2.resize(image, self.resize_shape)
                label = cv2.resize(label, self.resize_shape)

            # Do augmentation (only if training)
            if self.mode == 'train':
                if self.brightness:
                    factor = 1.0 + abs(random.gauss(mu=0.0, sigma=self.brightness))
                    if random.randint(0, 1):
                        factor = 1.0/factor
                    table = np.array([((i / 255.0) ** factor) *
                                      255 for i in np.arange(0, 256)]).astype(np.uint8)
                    image = cv2.LUT(image, table)

            if self.net == 'ICNET':
                self.X[n] = image
                self.Y1[n] = to_categorical(cv2.resize(label, (label.shape[1]//4, label.shape[0]//4)),
                                            self.n_classes).reshape((label.shape[0]//4, label.shape[1]//4, -1))
                self.Y2[n] = to_categorical(cv2.resize(label, (label.shape[1]//8, label.shape[0]//8)),
                                            self.n_classes).reshape((label.shape[0]//8, label.shape[1]//8, -1))
                self.Y3[n] = to_categorical(cv2.resize(label, (label.shape[1]//16, label.shape[0]//16)),
                                            self.n_classes).reshape((label.shape[0]//16, label.shape[1]//16, -1))
            else:
                self.X[n] = image
                self.Y[n] = to_categorical(cv2.resize(label, (label.shape[1], label.shape[0])),
                                           self.n_classes).reshape((label.shape[0], label.shape[1], -1))

        if self.net == 'ICNET':
            return self.X, [self.Y1, self.Y2, self.Y3]
        else:
            return self.X, self.Y

    def on_epoch_end(self):
        # Shuffle dataset for next epoch
        c = list(zip(self.image_path_list, self.label_path_list))
        random.shuffle(c)
        self.image_path_list, self.label_path_list = zip(*c)

        # Fix memory leak (Keras bug)
        gc.collect()


class PolyDecay:
    def __init__(self, initial_lr, power, n_epochs):
        self.initial_lr = initial_lr
        self.power = power
        self.n_epochs = n_epochs

    def scheduler(self, epoch):
        return self.initial_lr * np.power(1.0 - 1.0*epoch/self.n_epochs, self.power)

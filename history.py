#!/usr/bin/python3

import os
import matplotlib
import matplotlib.pyplot as plt


def plot_history(args, history, result_dir='history'):
    plt.plot(history.history['conv6_cls_categorical_accuracy'],
             marker='.')
    plt.plot(history.history['val_conv6_cls_categorical_accuracy'],
             marker='.')
    plt.title('model accuracy {}'.format(args.net))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['conv6_cls_categorical_accuracy', 'val_conv6_cls_categorical_accuracy'],
               loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy_{}.png'.format(args.net)))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss {}'.format(args.net))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss_{}.png'.format(args.net)))
    plt.close()


def save_history(args, history, result_dir='history'):
    loss = history.history['loss']
    acc = history.history['conv6_cls_categorical_accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_conv6_cls_categorical_accuracy']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result_{}.txt'.format(args.net)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()

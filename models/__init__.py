#!/usr/bin/python3

from .icnet import ICNET

def get_model(net, width, height, num_classes, weights_path=None, train=False):

    if net == 'ICNET':
        return ICNET(width, height, num_classes, weights_path, train)

    # elif net == '':
    #     return (width, height, num_classes, weights_path=None, train=False)

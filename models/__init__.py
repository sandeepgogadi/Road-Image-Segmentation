#!/usr/bin/python3

from .fcn8 import FCN8
from .icnet import ICNET
from .pspnet import PSPNET
from .segnet import SEGNET
from .unet import UNET


def get_model(args):

    width = args.width
    height = args.height
    num_classes = args.num_classes
    weights_path = args.weights_path
    train = args.train
    net = args.net

    if net == 'FCN8':
        return FCN8(width, height, num_classes, weights_path=None, train=False)
    elif net == 'ICNET':
        return ICNET(width, height, num_classes, weights_path=None, train=False)
    elif net == 'PSPNET':
        return PSPNET(width, height, num_classes, weights_path=None, train=False)
    elif net == 'SEGNET':
        return SEGNET(width, height, num_classes, weights_path=None, train=False)
    elif net == 'UNET':
        return UNET(width, height, num_classes, weights_path=None, train=False)

    # elif net == '':
    #     return (width, height, num_classes, weights_path=None, train=False)

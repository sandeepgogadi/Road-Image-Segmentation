#!/usr/bin/python3

from .fcn8 import FCN8
from .icnet import ICNET
from .pspnet import PSPNET
from .segnet import SEGNET
from .unet import UNET


def get_model(net, width, height, num_classes, weights_path=None, train=False):

    if net == 'ICNET':
        return ICNET(width, height, num_classes, weights_path, train)

    # elif net == '':
    #     return (width, height, num_classes, weights_path=None, train=False)

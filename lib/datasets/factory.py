# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
import datasets.kakou
import numpy as np

__sets = {}
imageset = 'KakouTrain'
devkit = '/home/bsl/KITTI-detection/data'


def get_imdb(name):
    """Get an imdb (image database) by name."""
    __sets['KakouTrain'] = (lambda imageset = imageset, devkit = devkit: datasets.kakou(imageset,devkit))
    __sets['KakouTest'] = (lambda imageset = 'KakouTest', devkit = devkit: datasets.kakou(imageset,devkit))
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

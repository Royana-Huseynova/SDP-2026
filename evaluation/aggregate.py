# Copyright (c) 2019 Luis F. Simoes (github: @lfsimoes)
# Licensed under the GPL License.

import numpy as np
import scipy.stats
import warnings
from data.io import lowres_image_iterator
from data.transforms import bicubic_upscaling


def baseline_upscale(path):
    """
    ESA competition baseline: bicubic upscale of clearest LR frames.
    """
    clearance = {}
    for (l, c) in lowres_image_iterator(path, img_as_float=True):
        clearance.setdefault(c.sum(), []).append(l)
    imgs = max(clearance.items(), key=lambda i: i[0])[1]
    sr = np.mean([bicubic_upscaling(i) for i in imgs], axis=0)
    return sr


def central_tendency(images, agg_with='median',
                     only_clear=False, fill_obscured=False,
                     img_as_float=True):
    """
    Aggregate LR frames with mean, median, or mode.
    """
    agg_opts = {
        'mean'  : lambda i: np.nanmean(i, axis=0),
        'median': lambda i: np.nanmedian(i, axis=0),
        'mode'  : lambda i: scipy.stats.mode(i, axis=0, nan_policy='omit').mode[0],
    }
    agg = agg_opts[agg_with]
    imgs = []
    obsc = []

    if isinstance(images, str):
        images = lowres_image_iterator(images, img_as_float or only_clear)
    elif only_clear:
        images = [(l.copy(), c) for (l, c) in images]

    for (l, c) in images:
        if only_clear:
            if fill_obscured != False:
                o = l.copy()
                o[c] = np.nan
                obsc.append(o)
            l[~c] = np.nan
        imgs.append(l)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        agg_img = agg(imgs)
        if only_clear and fill_obscured != False:
            if isinstance(fill_obscured, str):
                agg = agg_opts[fill_obscured]
            some_clear = np.isnan(obsc).any(axis=0)
            obsc = agg(obsc)
            obsc[some_clear] = 0.0
            np.nan_to_num(agg_img, copy=False)
            agg_img += obsc

    return agg_img
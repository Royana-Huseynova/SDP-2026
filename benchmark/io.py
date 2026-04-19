# 
# Copyright (c) 2019 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the GPL License. See the LICENSE file for details.

import os
from glob import glob
from zipfile import ZipFile
import numpy as np
import skimage
import skimage.io

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x

# [============================================================================]

def highres_image(path, img_as_float=True):
    """
    Load a scene's high resolution image and its corresponding status map.
    """
    path = path if path[-1] in {'/', '\\'} else (path + '/')

    hr = skimage.io.imread(path + 'HR.png')
    sm = skimage.io.imread(path + 'SM.png').astype(bool)

    if img_as_float:
        hr = skimage.img_as_float64(hr)

    return (hr, sm)


def lowres_image_iterator(path, img_as_float=True):
    """
    Iterator over all of a scene's low-resolution images.
    """
    path = path if path[-1] in {'/', '\\'} else (path + '/')

    for f in glob(path + 'LR*.png'):
        q = f.replace('LR', 'QM')

        l = skimage.io.imread(f)
        c = skimage.io.imread(q).astype(bool)

        if img_as_float:
            l = skimage.img_as_float64(l)

        yield (l, c)

# [============================================================================]

def check_img_as_float(img, validate=True):
    if not issubclass(img.dtype.type, np.floating):
        img = skimage.img_as_float64(img)

    if validate:
        assert img.min() >= 0.0 and img.max() <= 1.0

    return img

# [============================================================================]

def all_scenes_paths(base_path):
    base_path = base_path if base_path[-1] in {'/', '\\'} else (base_path + '/')
    return [
        base_path + c + s
        for c in ['RED/', 'NIR/']
        for s in sorted(os.listdir(base_path + c))
        if not s.startswith('.')
    ]


def scene_id(scene_path, incl_channel=False):
    sep = os.path.normpath(scene_path).split(os.sep)
    if incl_channel:
        return '/'.join(sep[-2:])
    else:
        return sep[-1]

# [============================================================================]

def prepare_submission(images, scenes, subm_fname='submission.zip'):
    assert len(images) == len(scenes), "Mismatch in number of images and scenes."
    assert subm_fname[-4:] == '.zip'

    print('Preparing submission. Writing to "%s".' % subm_fname)

    with ZipFile(subm_fname, mode='w') as zf:
        for img, scene in zip(tqdm(images), scenes):
            skimage.io.imsave('tmp.png', img)
            zf.write('tmp.png', arcname=scene_id(scene) + '.png')

    if os.path.exists('tmp.png'):
        os.remove('tmp.png')
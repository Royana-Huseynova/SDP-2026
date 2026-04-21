# 
# Copyright (c) 2019 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the GPL License. See the LICENSE file for details.

import skimage

# [============================================================================]

def bicubic_upscaling(img):
    """
    Compute a bicubic upscaling by a factor of 3.
    NOTE: multichannel=False was removed as it is deprecated in modern skimage.
    """
    r = skimage.transform.rescale(img, scale=3, order=3, mode='edge',
                                  anti_aliasing=False)
    return r
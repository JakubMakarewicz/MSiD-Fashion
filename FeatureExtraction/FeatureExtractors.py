import cv2
import numpy as np
from skimage import filters, feature
from functools import partial
import torch
from skimage.util import random_noise

kernel = [((3, 3), 5, 3.9269908169872414, 0.7853981633974483, 0.5, 0.4), None]

feature_extractors = {"prewitt_h": filters.prewitt_h,
                      "prewitt_v": filters.prewitt_v,
                      "sobel":     filters.sobel,
                      "canny":     feature.canny,
                      "gabor":     partial(cv2.filter2D, ddepth=cv2.CV_8UC3,
                                           kernel=cv2.getGaborKernel(*kernel[0], ktype=cv2.CV_32F))}


def initialize_kernel(reset, *params):
    if reset:
        kernel[0] = ((3, 3), 5, 3.9269908169872414, 0.7853981633974483, 0.5, 0.4)
    else:
        kernel[0] = params

def apply_noise(images):
    images = np.array(images).astype('float32')
    ret_array = []
    for image in images:
        image = torch.tensor(random_noise(image, mode='gaussian', mean=0, var=0.003, clip=True))
        ret_array.append(np.array(image))
    return ret_array


def feature_extraction(images, method=None):
    ret_array = []
    for image in images:

        image = np.reshape(np.array(image), (28, 28))
        if method is not None:
            image = feature_extractors[method](image)
        ret_array.append(image)
    return ret_array

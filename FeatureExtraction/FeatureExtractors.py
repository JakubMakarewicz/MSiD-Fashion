import numpy as np
from skimage import filters, feature

feature_extractors = {"prewitt_h": filters.prewitt_h,
                      "prewitt_v": filters.prewitt_v,
                      "sobel":     filters.sobel,
                      "canny":     feature.canny}

def feature_extraction(images, method):
    ret_array = []
    for image in images:
        # image = np.reshape(image, (28, 28))
        ret_array.append(feature_extractors[method](image))
    return ret_array

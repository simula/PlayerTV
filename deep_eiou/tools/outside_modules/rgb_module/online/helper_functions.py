import numpy as np
import cv2

from skimage import color

def weighted_RGB(crop: np.array):
    weights = np.array([0.3, 0.59, 0.11])
    crop = crop * weights
    value = cv2.mean(crop)
    value = value[:3]
    return np.asarray(value)


def to_cielab(crop: np.array):
    image_lab = color.rgb2lab(crop)
    value = cv2.mean(image_lab)
    value = value[:3]
    return np.asarray(value)


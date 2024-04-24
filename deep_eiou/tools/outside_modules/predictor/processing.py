import torch
import numpy as np
import cv2

def preprocess(img: np.array, input_size: tuple, rgb_mean: tuple, std: tuple, swap = (2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114
    else:
        padded_img = np.ones(input_size) * 114

    img = np.array(img)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation = cv2.INTER_LINEAR,
    ).astype(np.float32)

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    ##BGR to RGB
    #padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


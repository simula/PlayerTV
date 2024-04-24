import skvideo.measure as measure
import cv2
import numpy as np


def niqe(crop: np.array):
    def pad_image_to_size(crop):
        (target_h, target_w) = (194, 194)
        h, w = crop.shape[:2]
        delta_w = max(0, target_w - w)
        delta_h = max(0, target_h - h)
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]  # Black padding
        new_image = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return new_image
    
    if crop.shape[0] < 194 or crop.shape[1] < 194:
        print("alter")
        crop = pad_image_to_size(crop)
    print(crop.shape)
    grayscale_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    print(grayscale_crop.shape)
    score = measure.niqe(grayscale_crop)
    score = score[0]
    return score






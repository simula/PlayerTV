import numpy as np
from fractions import Fraction


def parse_offset(offset):
    for key, val in offset.items():
        if isinstance(val, str):
            offset[key] = float(sum(Fraction(s) for s in val.split()))
        else:
            offset[key] = val

    return offset


def get_crop_center_offset(crop: np.array, offset: dict):
    h, w, _ = crop.shape
    area = offset["area"]
    aspect_ratio = offset["aspect_ratio"]

    crop_width = min(w, int(area * aspect_ratio**0.5))
    crop_height = min(h, int(area / aspect_ratio**0.5))

    ## Coordinates of crop
    left = (w - crop_width) // 2
    top = (h - crop_height) // 2
    right = left + crop_width
    bot = top + crop_height

    crop = crop[top:bot, left:right]
    return crop

def get_crop_trim_offset(crop: np.array, offset: dict):
    h, w, _ = crop.shape
    off_y_top = offset["y_top"]
    off_y_bot = offset["y_bot"]
    off_x_left = offset["x_left"]
    off_x_right = offset["x_right"]

    y = int(off_y_top*h)
    y_hat = int(h - off_y_bot*h)
    x = int(off_x_left*w)
    x_hat = int(w - off_x_right*w)

    y = np.clip(y, 0, h)
    y_hat = np.clip(y_hat, 0, h)
    x = np.clip(x, 0, w)
    x_hat = np.clip(x_hat, 0, w)

    crop = crop[y:y_hat, x:x_hat]

    return crop

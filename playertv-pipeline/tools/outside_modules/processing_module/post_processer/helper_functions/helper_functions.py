import cv2
import numpy as np

def split_into_equal_parts(arr, num_parts):
    avg = len(arr)/float(num_parts)
    last = 0.0
    out = []
    while last < len(arr):
        out.append(arr[int(last): int(last + avg)])
        last += avg
    return out
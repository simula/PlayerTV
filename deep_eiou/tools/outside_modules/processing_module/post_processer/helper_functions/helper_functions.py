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


def get_frame_by_number(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        return frame
    else:
        return None
    

def get_crop_from_frame(frame, coord):
    x, y, w, h = coord
    y = int(y)
    x = int(x)
    w = int(w)
    h = int(h)
    
    y = np.clip(y, 0, frame.shape[0])
    y_hat = np.clip(y + h, 0, frame.shape[0])
    x = np.clip(x, 0, frame.shape[1])
    x_hat = np.clip(x+w, 0, frame.shape[1])

    crop = frame[y:y_hat, x:x_hat]
    return crop
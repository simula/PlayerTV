import cv2
from collections import defaultdict

from ..helper_functions.helper_functions import split_into_equal_parts

def split_into_time_zones(post_processor, crop_scores):
    num_of_time_zones = post_processor.num_of_time_zones
    time_zones = split_into_equal_parts(crop_scores, num_of_time_zones)
    return_list = []

    for time_zone in time_zones:
        sorted_time_zone = sorted(time_zone, key = lambda entry: entry["crop_score"])
        return_list.append(sorted_time_zone)
    return return_list


def top_n_scores(post_processor, crop_scores):
    n = post_processor.n
    return_list = []
    sorted_crop_scores = sorted(crop_scores, key = lambda entry: entry["crop_score"])
    return_list.append(sorted_crop_scores[:n])
    return return_list

def brute_force(post_processor, crop_scores):
    return_list = []
    sorted_crop_scores = sorted(crop_scores, key = lambda entry: entry["crop_score"])
    return_list.append(sorted_crop_scores)
    return return_list





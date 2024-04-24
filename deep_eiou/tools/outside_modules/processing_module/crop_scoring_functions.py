from .post_processer.smart_crop_extractor import Sample_Crop
from brisque import BRISQUE
brisque = BRISQUE(url=False)

def iou_score(crop, iou):
    score = brisque.score(crop)

    score = (iou * score) + score
    return score

def iou_area(crop, iou):
    w, h, _ = crop.shape
    area = w * h
    return 1/area + iou
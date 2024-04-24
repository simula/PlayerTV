import numpy as np
from collections import defaultdict

class Sample_Crop:
    def __init__(self, txt_line):
        ## Set parameters
        info = txt_line.strip().split(",")

        self.frame = int(info[0])
        self.id = int(info[1])
        self.xywh = (float(info[2]), float(info[3]), float(info[4]), float(info[5]))
        self.conf = float(info[6])

        rgb_value = info[7]
        rgb_value = rgb_value.strip("[ ]")
        self.avg_rgb = np.fromstring(rgb_value, dtype=int, sep=" ")

        self.brisque = float(info[8])
        self.iou = float(info[9])

        self.score = (self.iou * self.brisque) + self.brisque

        self.team = None
        self.ocr = None
        if len(info) > 10:
            self.team = info[10]
            self.ocr = info[11]

    
    def set_ocr(self, value):
        self.ocr = value

    def set_team(self, value):
        self.team = value



class Smart_Crop_Detector:
    def __init__(self, score_threshold = 0.5, max_samples = 10):
        self.score_threshold = score_threshold
        self.max_samples = max_samples

        #ID is key and list of samples is value. List is sorted in ascending order based on score
        self.good_frames = defaultdict(lambda: np.full(self.max_samples, None, dtype=object))
        self.good_frames_scores = defaultdict(lambda: np.full(self.max_samples, None, dtype=float))

    def return_samples(self):
        return self.good_frames
    
    def reset_samples(self):
        self.good_frames = None
        self.good_frames_scores = None
        self.good_frames = defaultdict(lambda: np.full(self.max_samples, None, dtype=object))
        self.good_frames_scores = defaultdict(lambda: np.full(self.max_samples, None, dtype=float))

    def _insert_sample(self, sample_crop, scores_list, good_frames_list):
        score = sample_crop.score

        insert_index = np.searchsorted(scores_list, score)

        scores_list[insert_index+1:] = scores_list[insert_index:-1]
        good_frames_list[insert_index+1:] = good_frames_list[insert_index:-1]

        scores_list[insert_index] = score
        good_frames_list[insert_index] = sample_crop

        return scores_list, good_frames_list
    
    def _insert_sample_player(self, frame_dict, scores_list, good_frames_list):
        score = (frame_dict["brisque_score"] * frame_dict["iou"]) + frame_dict["brisque_score"]

        insert_index = np.searchsorted(scores_list, score)

        scores_list[insert_index+1:] = scores_list[insert_index:-1]
        good_frames_list[insert_index+1:] = good_frames_list[insert_index:-1]

        scores_list[insert_index] = score
        good_frames_list[insert_index] = frame_dict

        return scores_list, good_frames_list
    

    def check_sample(self, sample_crop : Sample_Crop):
        score = sample_crop.score
        id = sample_crop.id

        good_frames_list = self.good_frames[id]
        scores_list = self.good_frames_scores[id]
        if np.isnan(scores_list[0]):
            scores_list[0] = score
            good_frames_list[0] = sample_crop
            
        else:
            worst_score = scores_list[-1]
            if score < worst_score or np.isnan(worst_score):
                scores_list, good_frames_list = self._insert_sample(sample_crop, scores_list, good_frames_list)

    def check_sample_player(self, frame_dict):
        score = (frame_dict["brisque_score"] * frame_dict["iou"]) + frame_dict["brisque_score"]
        id = frame_dict["track_id"]

        good_frames_list = self.good_frames[id]
        scores_list = self.good_frames_scores[id]
        if np.isnan(scores_list[0]):
            scores_list[0] = score
            good_frames_list[0] = frame_dict
            
        else:
            worst_score = scores_list[-1]
            if score < worst_score or np.isnan(worst_score):
                self.good_frames_scores[id], self.good_frames[id] = self._insert_sample_player(frame_dict, scores_list, good_frames_list)

    

        





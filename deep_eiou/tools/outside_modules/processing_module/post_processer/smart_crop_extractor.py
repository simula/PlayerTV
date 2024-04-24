import numpy as np
from collections import defaultdict

class Sample_Crop:
    def __init__(self, sample):
        self.team = None
        self.ocr = None
        self.score = None
        if isinstance(sample, str):
            ## Set parameters
            info = sample.strip().split(",")

            self.frame = int(info[0])
            self.id = int(info[1])
            self.xywh = (float(info[2]), float(info[3]), float(info[4]), float(info[5]))
            self.conf = float(info[6])

            rgb_value = info[7]
            rgb_value = rgb_value.strip("[ ]").split()
            self.avg_rgb = np.array(rgb_value, dtype=float)
            self.avg_rgb = self.avg_rgb.astype(int)

            self.brisque = float(info[8])
            self.iou = float(info[9])

            if len(info) > 10:
                self.team = info[10]
                self.ocr = info[11]

        elif isinstance(sample, dict):
            self.frame = sample["frame"]
            self.id = sample["track_id"]
            self.xywh = sample["xywh"]
            self.conf = sample["conf"]
            self.avg_rgb = sample["avg_rgb"]
            self.brisque = sample["crop_score"]
            self.iou = sample["iou"]

            if "team" in sample:
                self.team = sample["team"]
            if "ocr" in sample:
                self.ocr = sample["ocr"]
            

        else:
            raise NotImplementedError(f"Input sample not of acceptable type (str of dict). Your input type: {type(sample)}")

    
    def set_ocr(self, value):
        self.ocr = value

    def set_team(self, value):
        self.team = value

    def set_score(self, score):
        self.score = score



class Smart_Crop_Extractor:
    def __init__(self, crop_score_function, score_threshold = 0.5, max_samples = 10):
        self.score_threshold = score_threshold
        self.max_samples = max_samples

        self.score_function = crop_score_function

        #ID is key and list of samples is value. List is sorted in ascending order based on score
        # Low score is the better score
        self.good_frames = defaultdict(lambda: np.full(self.max_samples, None, dtype=object))
        self.good_frames_scores = defaultdict(lambda: np.full(self.max_samples, None, dtype=float))

    def return_samples(self):
        ## Returning the top scoring smaples for each id with maximum samples size
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
    

    def check_sample(self, sample_crop : Sample_Crop):
        score = self.score_function(sample_crop)
        sample_crop.set_score(score)
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

    
    def get_sorted_list_of_samples_scores(self, sample_list: list[dict]):
        ## This function assumes a list belonging to a single track id. Sorting all entriers given, treating as one tracklet
        sorted_sample_list = np.full(len(sample_list), None, dtype=object)
        sorted_score_list = np.full(len(sample_list), None, dtype=float)

        for sample in sample_list:
            sample = Sample_Crop(sample)
            if sample.score:
                score = sample.score
            else:
                score = self.score_function(sample)
                sample.set_score(score)

            if np.isnan(sorted_score_list[0]):
                sorted_score_list[0] = score
                sorted_sample_list[0] = sample

            else:
                worst_score = sorted_score_list[-1]
                if score < worst_score or np.isnan(worst_score):
                    insert_index = np.searchsorted(sorted_score_list, score)
                    sorted_score_list[insert_index+1:] = sorted_score_list[insert_index: -1]
                    sorted_sample_list[insert_index+1:] = sorted_sample_list[insert_index: -1]

                    sorted_score_list[insert_index] = score
                    sorted_sample_list[insert_index] = sample

        return sorted_sample_list, sorted_score_list




    

        





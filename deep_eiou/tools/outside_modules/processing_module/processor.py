import os.path as osp
from typing import Union
import re
import json
import numpy as np
from collections import defaultdict

from get_metadata.get_metadata import resolve_color


from .crop_scoring_functions import iou_score
from .ocr_scoring_functions import highest_in_player_list
from .post_processer.smart_crop_extractor import Sample_Crop
from .read_results_from_file import OnlineTargetFile, OnlineTargetLoader

crop_scoring_functions = {"iou_score": iou_score}
ocr_scoring_functions = {"highest_in_player_list": highest_in_player_list}

class Processor:
    def __init__(self, parameters, rgb_module, ocr_module, predictor):
        self.cap = None
        self.path = parameters["path"]
        self.parameters = parameters

        self.ocr_score_function = ocr_scoring_functions[parameters["ocr_score_function"]]
        try:
            self.crop_score_function = crop_scoring_functions[parameters["crop_score_function"]]
        except:
            raise NotImplementedError(f"Scoring function with name '{parameters['params']['crop_score_function']}' not implemented. Check your config file")

        self.rgb_module = rgb_module
        self.ocr_module = ocr_module
        self.predictor = predictor

        self.stop_at_detection = parameters["stop_at_detection"]

        self.all_ocr_results = None
        self.cluster_map = None
        self.online_target_loader = None
        self.final_json = {}
        self.team_player_dict = defaultdict(lambda: defaultdict(lambda: None))
        self.player_detections = defaultdict(lambda: [])
        self.id_player_mapping = defaultdict(lambda: {"team": None, "kit_number": -1, "name": "NA"})
        self.id_player_mapping = {}
        self.pre_processed_results = defaultdict(lambda: [])

        ## Online tracklets mapping using team idx and kit number detected
        self.team_kit_numbers = {0: defaultdict(lambda: None), 1: defaultdict(lambda: None)}

    def get_metadata(self, txt, cluster0=[0,0,0], cluster1=[0,0,0], cluster = False):
        pattern = r"(\d{4})_[A-Za-z0-9]+"
        a = re.search(pattern, txt)

        if not a:
            return False
        game_id = int(a.group(1))
        if cluster:
            rgb_module = self.rgb_module
        else:
            rgb_module = None
        api_metadata = resolve_color(game_id, cluster0, cluster1, api_info=None, rgb_module=rgb_module)
        self.final_json["metadata"]["api_info"] = api_metadata
        return True

    def process_frame(self, online_targets, frame, frame_id, iou_matrix):
        raise NotImplementedError("Subclasses must implement process_frame method.")
    
    def post_process(self):
        raise NotImplementedError("Subclasses must implement post_process method.")
    
    def get_crop_from_frame(self, frame, tlwh):
        x, y, w, h = tlwh
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        
        y = np.clip(y, 0, frame.shape[0])
        y_hat = np.clip(y + h, 0, frame.shape[0])
        x = np.clip(x, 0, frame.shape[1])
        x_hat = np.clip(x+w, 0, frame.shape[1])

        crop = frame[y:y_hat, x:x_hat]
        return crop
    
    def iou_sum(self, iou_matrix, index):
        row_sum = np.sum(iou_matrix[index, index+1:])
        col_sum = np.sum(iou_matrix[:index, index])
        return row_sum + col_sum

    def set_metadata(self, metadata):
        self.final_json["metadata"] = metadata

    def load_tracker_results(self, tracker_results):
        target_loader = OnlineTargetLoader(tracker_results)
        self.online_target_loader = target_loader

    def load_ocr_detections(self, ocr_path):
        with open(ocr_path, "r", encoding="utf-8") as file:
            self.all_ocr_results = json.load(file)
        
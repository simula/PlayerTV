from collections import defaultdict
from typing import Union
import json
import os.path as osp
import cv2
import numpy as np

from ..processor import Processor
from .smart_crop_extractor import Sample_Crop, Smart_Crop_Extractor
from .helper_functions.get_ocr_results import split_into_time_zones
from .helper_functions.decide_kit_number import highest_in_player_list
from .helper_functions.decide_team_scoring_functions import plurality_vote, min_total_distance
from ..crop_scoring_functions import iou_area, iou_score

get_ocr = {"split_into_time_zones": split_into_time_zones, "brute_force": None, "top_n_crops": None}
decide_ocr = {"highest_in_player_list": highest_in_player_list}
decide_team_scores = {"min_total_distance": min_total_distance, "plurality_vote": plurality_vote}
crop_score_functions = {"iou_score": iou_score, "iou_area": iou_area}




class PostProcessor(Processor):
    def __init__(self, parameters, rgb_module, ocr_module, predictor):
        super().__init__(parameters, rgb_module, ocr_module, predictor)
        try:
            self.num_of_time_zones = parameters["params"]["num_of_time_zones"]
        except:
            pass

        smart_crop_params = parameters["params"]["smart_crop_params"]
        self.smart_crop_extractor = Smart_Crop_Extractor(crop_score_function = self.crop_score_function, **smart_crop_params)

        ocr_function = parameters["params"]["get_ocr_function"]
        self.get_ocr_function = get_ocr[ocr_function]
        decide_ocr_function = parameters["params"]["decide_ocr_function"]
        self.final_ocr_scoring_function = decide_ocr[decide_ocr_function]
        decide_team_score_function = parameters["decide_team_scores"]
        self.decide_team_function = decide_team_scores[decide_team_score_function]

        self.id_mapping_rgb = defaultdict(lambda: [])
        self.crop_scores = defaultdict(lambda: [])
        self.id_mapping_ocr = defaultdict(lambda: [])
        self.all_tracklets = defaultdict(lambda: None)

    def process_frame(self, online_targets, frame, frame_id, iou_matrix):
        for i, t in enumerate(online_targets):
            if self.all_tracklets[t.track_id] == None:
                self.all_tracklets[t.track_id] = {"start_frame": frame_id}
            self.all_tracklets[t.track_id]["end_frame"] = frame_id
            sum_iou = self.iou_sum(iou_matrix, i)
            avg_color = -1

            if sum_iou < self.rgb_module.certain_iou_thresh or len(self.crop_scores[t.track_id]) == 0:
                crop = self.get_crop_from_frame(frame, t.tlwh)
                avg_color = self.rgb_module.get_avg_rgb(crop)
                score = self.crop_score_function(crop, sum_iou)
                score_entry = {                     
                    "frame_id": frame_id,
                    "tlwh": t.tlwh,
                    "crop_score": score,
                    "avg_color": avg_color
                }
                self.crop_scores[t.track_id].append(score_entry)
                self.id_mapping_rgb[t.track_id].append(avg_color)
                if sum_iou < self.rgb_module.certain_iou_thresh:
                    self.rgb_module.add_sample(avg_color)
            
            tracker_entry = {
                t.track_id: {
                    "frame": frame_id,
                    "track_id": t.track_id,
                    "xywh": t.tlwh,
                    "conf": t.score,
                }
            }
            self.pre_processed_results[frame_id].append(tracker_entry)

    def post_process(self):
        cluster_map = self.get_cluster_map()
        for track_id, crop_scores in self.crop_scores.items():
            found_ocr = False
            color_samples = self.id_mapping_rgb[track_id]
            team_id = self.decide_team_function(self, color_samples)
            player_list = None
            if cluster_map:
                team_name = cluster_map[team_id]
                if team_name == self.final_json["metadata"]["api_info"]["home_team"]:
                    player_list = self.final_json["metadata"]["api_info"]["home_team_player"]

                elif team_name == self.final_json["metadata"]["api_info"]["away_team"]:
                    player_list = self.final_json["metadata"]["api_info"]["away_team_player"]
            
            ocr_crops = self.get_ocr_function(self, crop_scores)
            accumulator_dict = defaultdict(int)
            for time_zone in ocr_crops:
                skip_time_zone = False
                for sample in time_zone:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, sample["frame_id"])
                    ret, frame = self.cap.read()
                    crop = self.get_crop_from_frame(frame, sample["tlwh"])
                    txts, boxes, conf = self.ocr_module.run_ocr(crop)
                    for i in range(len(txts)):
                        found_ocr = True
                        txt = txts[i]
                        box = boxes[i]
                        confidence = conf[i]
                        accumulator_dict[txt] += 1
                        accumulator = accumulator_dict[txt]
                        
                        total_score, player_txt = self.ocr_score_function(txt, box, confidence, accumulator, crop, player_list)
                        entry = {"track_id": track_id, "player_name": player_txt, "kit_num": int(txt), "team": team_name, "team_idx": team_id, "score": total_score}
                        self.id_mapping_ocr[track_id].append(entry)

                        if self.stop_at_detection:
                            skip_time_zone = True

                    if skip_time_zone:
                        break
            if not found_ocr:
                team = team_id
                if player_list:
                    team = team_name
                self.id_player_mapping[track_id] = {"team": team, "kit_number": -1, "name": "NA"}


        ### CREATE ID PLAYER MAPPING; ALSO CREATE RESULTS: PREPROCESSED TRACKER; FULL INFO; AND PLAYER ID MAP(TEAMNAME AND KIT NUM)
        self.create_final_results()
        self.save_results()

    def get_cluster_map(self):
        clusters = self.rgb_module.cluster()
        
        if self.get_metadata(self.path, clusters[0], clusters[1], True):
            api_info = self.final_json["metadata"]["api_info"]
            home_team_idx = int(api_info["home_team_idx"])
            away_team_idx = int(api_info["away_team_idx"])
            home_team = api_info["home_team"]
            away_team = api_info["away_team"]
            self.cluster_map = {home_team_idx: home_team, away_team_idx: away_team}
            print(self.cluster_map)
            return self.cluster_map
            
        return None
    
    def create_final_results(self):
        for track_id, values in self.id_mapping_ocr.items():
            values.sort(key=lambda entry: entry["score"], reverse=True)
            top_score = values[0]
            team = top_score["team"]
            kit_num = top_score["kit_num"]
            team_id = top_score["team_idx"]
            if self.team_kit_numbers[team_id][kit_num] == None:  
                self.team_kit_numbers[team_id][kit_num] = [top_score]
                self.team_player_dict[track_id] = {"team": team, "kit_number": kit_num, "name": top_score["player_name"]}
            else:
                self.resolve_conflict(track_id, self.team_kit_numbers[team_id][kit_num], team_id, kit_num)
        kit_nums = dict(self.team_player_dict)
        teams = dict(self.id_player_mapping)
        final_res = defaultdict(lambda: [])
        for frame, values in self.pre_processed_results.items():
            for value in values:
                for track_id, details in value.items():
                    if track_id in kit_nums:
                        myentry = kit_nums[track_id]
                    else:
                        myentry = teams[track_id]

                    entry = {track_id: {
                        "frame": details["frame"],
                        "track_id": details["track_id"],
                        "xywh": details["xywh"],
                        "conf": details["conf"],
                        "team_id": myentry["team"],
                        "kit_number": myentry["name"]
                    }}
                    final_res[frame].append(entry)
        self.final_json["frame"]=final_res


    def save_results(self):
        from helper_functions import save_file
        final = self.final_json
        path = osp.join("results")
        save_file(final, path, "final_results", "json")





    def resolve_conflict(self, track_ida, track_idb_list, team_id, kit_num):
        team = team_id
        if self.cluster_map:
            team = self.cluster_map[team_id]
        for track in track_idb_list:
            track_idb = track["track_id"]
            ##IDA is the challenging tracklet
            entry1 = self.all_tracklets[track_ida]
            entry2 = self.all_tracklets[track_idb]
            overlap1 = entry1['start_frame'] <= entry2['end_frame'] and entry1['end_frame'] >= entry2['start_frame']
            overlap2 = entry2['start_frame'] <= entry1['end_frame'] and entry2['end_frame'] >= entry1['start_frame']
            valuesa = self.id_mapping_ocr[track_ida]
            valuesa.sort(key=lambda entry: entry["score"], reverse = True)
            if overlap1 or overlap2:
                ## They are within the same time segment, further comparisons must be made
                print(track_ida, track_idb)
                valuesb = self.id_mapping_ocr[track_idb]
                valuesb.sort(key=lambda entry: entry["score"], reverse = True)
                strip_list = valuesa
                top_list = valuesb
                if valuesa[0]["score"] > valuesb[0]["score"]:
                    strip_list = valuesb
                    top_list = valuesa
                id_strip_list = strip_list[0]["track_id"]
                while len(strip_list) > 0 and strip_list[0]["kit_num"] == top_list[0]["kit_num"]:
                    del strip_list[0]

                if len(strip_list) == 0:
                    self.team_player_dict[id_strip_list] = {"team": team, "kit_number": -1, "name": "NA"}
                    self.id_player_mapping[id_strip_list] = {"team": team, "kit_number": -1, "name": "NA"}
                    return
                else:
                    team_id = strip_list[0]["team_idx"]
                    kit_num = strip_list[0]["kit_num"]
                    track_id = strip_list[0]["track_id"]
                    if self.team_kit_numbers[team_id][kit_num] == None:  
                        self.team_kit_numbers[team_id][kit_num] = [strip_list[0]]
                        self.team_player_dict[track_id] = {"team": strip_list[0]["team"], "kit_number": kit_num, "name": strip_list[0]["player_name"]}
                    else:
                        return self.resolve_conflict(id_strip_list, self.team_kit_numbers[team_id][kit_num]["track_id"], team_id, kit_num)
            else:
                self.team_player_dict[track_ida] = {"team": team, "kit_number": kit_num, "name": valuesa[0]["player_name"]}
                self.team_kit_numbers[team_id][kit_num].append(valuesa[0])
                return


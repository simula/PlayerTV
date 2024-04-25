from collections import defaultdict
import numpy as np

possible_states = ["certain", "uncertain", "danger_zone"]

class OnlineTracklet:
    certain_iou_thresh = 0.1
    danger_zone_iou_thresh = 0.5
    conf_thresh = 0.5
    team_id_conf_thresh = 0.75
    ocr_confirmation_lim = 3
    team_id_confirmation_lim = 10

    def __init__(self, track_id, frame_num, iou):
        self.id = track_id
        self.original_id = track_id
        self.state = None
        self.post_dz = False
        self.ocr_detection = None
        self.rgb = None
        self.team_id = None
        self.iou = None
        self.num_of_dz = 0
        self.dz = None
        self.taken_index = 0
        self.team_change = 0
        self.rgb_values = []
        self.name = None

        self.rgb_mean = np.array([0.0, 0.0, 0.0])
        self.rgb_running_mean_count = 0

        self.detection_dicts = {state: defaultdict(lambda: 0) for state in possible_states}
        self.ocr_detections = {state: np.zeros(10, dtype=int) for state in possible_states}
        self.ocr_scores = {state: np.zeros(10, dtype=np.float64) for state in possible_states}

        self.team_counts = {state: {0:0, 1:0} for state in possible_states}

        self.detection_dict_post_dz = defaultdict(lambda: 0)
        self.ocr_detections_pre_dz = np.zeros(10)
        self.ocr_scores_post_dz = np.zeros(10)
        self.team_count_post_dz = {0:0, 1:0}
        self.team_count_pre_dz = {0:0, 1:0}

        self.start_frame = frame_num
        self.last_frame = frame_num
        self.update_state(iou, frame_num)

    def __str__(self) -> str:
        return(
            f"ID: {self.id}, TEAM: {self.get_team_id()}, Kit: {self.get_ocr()}, NUM OF DZ: {self.num_of_dz}. Active from {self.start_frame} - {self.last_frame}"
        )
    
    def set_ocr(self, ocr):
        self.ocr_detection = ocr, 
    
    def update_state(self, iou, frame_num):
        self.iou = iou
        self.last_frame = frame_num
        self.taken_index = 0
        self.team_change = False
        if iou < self.certain_iou_thresh:
            self.state = "certain"
        elif iou < self.danger_zone_iou_thresh:
            self.state = "uncertain"
        else:
            self.state = "danger_zone"

    def check_team_id_overweight(self):
        total_count = sum(self.team_counts["certain"].values())
        if total_count == 0:
            return False
        for cluster, count in self.team_counts["certain"].items():
            percentage = count/total_count
            if percentage > self.team_id_conf_thresh:
                self.team_id = cluster
                return True
        return False
    
    def is_certain(self):
        if self.ocr_detections["certain"][-1] != 0:
            self.ocr_detection = self.ocr_detections["certain"][-1]
            return True
        if self.check_team_id_overweight():
            return True
        return False
    
    def ocr_is_consistent(self, txt=None, other_track=None):
        if other_track == self:
            other_track=None
        ocr_detections_pre_dz = self.ocr_detections_pre_dz
        detection_dict_post_dz = self.detection_dict_post_dz
        if other_track:
            ocr_detections_pre_dz = other_track.ocr_detections_pre_dz
            detection_dict_post_dz = other_track.detection_dict_post_dz

        if txt:
            txt = int(txt)
            if not other_track:
                detection_dict_post_dz[txt] += 1

            num_of_detections = detection_dict_post_dz[txt]
            if num_of_detections >= self.ocr_confirmation_lim:
                if txt in ocr_detections_pre_dz:
                    return True
        else:
            txt = max(detection_dict_post_dz, key=detection_dict_post_dz.get)
            txt = int(txt)
            if txt in ocr_detections_pre_dz:
                return True
        return False

    def rgb_is_consistent(self, cluster_id=None, other_track=None):
        if other_track == self:
            other_track=None
        team_count_pre_dz = self.team_count_pre_dz
        team_count_post_dz = self.team_count_post_dz
        if other_track:
            team_count_pre_dz = other_track.team_count_pre_dz
            team_count_post_dz = other_track.team_count_post_dz

        if cluster_id:
            if not other_track:
                team_count_post_dz[cluster_id] += 1
            my_cluster = team_count_post_dz[cluster_id]
            other_cluster = team_count_post_dz[1 - cluster_id]
            diff = my_cluster - other_cluster
            if diff >= self.team_id_confirmation_lim:
                pre_dz_cluster = max(team_count_pre_dz, key = team_count_pre_dz.get)
                if cluster_id == pre_dz_cluster:
                    return True
            
        else:
            pre_dz_cluster = max(team_count_pre_dz, key = team_count_pre_dz.get)
            post_dz_cluster = max(team_count_post_dz, key = team_count_post_dz.get)
            diff = team_count_post_dz[post_dz_cluster] - team_count_post_dz[1-post_dz_cluster]
            
            if diff < self.team_id_confirmation_lim:
                return False
            if post_dz_cluster == pre_dz_cluster:
                return True
            
        return False


    def run_ocr(self):
        if self.state == "certain":
            return True
        elif self.get_ocr() == None:
            return True
        elif self.state == "danger_zone":
            return False
        elif self.state == "uncertain":
            if self.ocr_detections["certain"][-1] != 0:
                return False
            return True
        
    def run_rgb(self):
        if self.state == "certain":
            return True
        elif self.get_rgb() == None:
            return True
        elif self.state == "danger_zone":
            return False
        elif self.state == "uncertain":
            return not self.check_team_id_overweight()


    def enter_dz(self, dz):
        self.dz = dz
        self.post_dz = True
        self.num_of_dz += 1
        self.team_count_pre_dz = self.team_counts["certain"].copy()
        self.ocr_detections_pre_dz = self.ocr_detections["certain"].copy()
        self.detection_dict_post_dz = defaultdict(lambda: 0)
        self.team_count_post_dz = {0:0, 1:0}

    def in_player_list(self, detection, player_list):
        for player in player_list:
            shirt_num = int(player["shirt_number"])
            if detection == shirt_num:
                self.name = f"{player['name']} ({shirt_num})"
                return True
        return False

    def add_ocr_detection(self, detection, score, player_list=None):
        if detection == None or detection == "":
            return
        if score < self.conf_thresh:
            return
        detection = int(detection)
       
        scores = self.ocr_scores[self.state]
        detections = self.ocr_detections[self.state]
        accumulator = self.detection_dicts[self.state]

        
        accumulator[detection] += 1
        score = score * accumulator[detection]
        if player_list:
            if self.in_player_list(detection, player_list):
                score = score * 10
        if score >= scores[-1]:
            scores[:-1] = scores[1:]
            scores[-1] = score
            detections[:-1] = detections[1:]
            detections[-1] = detection
        elif score <= scores[0]:
            return
        else:
            idx = np.searchsorted(scores, score, side="left") - 1
            if idx == 0:
                scores[idx] = score
                detections[idx] = detection
            else:
                scores[:idx] = scores[1:idx+1]
                scores[idx] = score
                detections[:idx] = detections[1:idx+1]
                detections[idx] = detection
       
    


    def add_rgb(self, rgb_value, cluster_id):
        team_count = self.team_counts[self.state]
        if self.state == "certain":
            self.update_running_mean(rgb_value)
            self.rgb = rgb_value
            self.rgb_values.append(rgb_value)
        
        team_count[cluster_id] += 1
        self.team_id = self.get_team_id()

    def update_running_mean(self, new_entry):
        self.rgb_running_mean_count += 1
        adjustment = (new_entry - self.rgb_mean) / self.rgb_running_mean_count
        self.rgb_mean += adjustment

    def get_team_id(self):
        self.get_rgb()
        return self.team_id

    def get_team_count(self):
        for state in possible_states:
            team_count = self.team_counts[state]
            if not all(value == 0 for value in team_count.values()):
                return (team_count, state)
    
    def get_ocr_top5(self):
        top5 = {"detections": [], "scores": [], "states": []}
        found  = 0
        for state in possible_states:
            idx = 0
            det = self.ocr_detections[state][-1-idx]
            while det != 0:
                if det not in top5["detections"]:
                    top5["detections"].append(det)
                    top5["scores"].append(self.ocr_scores[state][-1-idx])
                    top5["states"].append(state)
                    found += 1
                    idx += 1
                    if found == 5:
                        return top5, found
                    try:
                        det = self.ocr_detections[state][-1-idx]
                    except:
                        break
                else:
                    try:
                        idx += 1
                        det = self.ocr_detections[state][-1-idx]
                    except:
                        break
                
        return top5, found


    def get_ocr(self):
        for state in possible_states:
            detection =  self.ocr_detections[state][-1]
            if detection != 0:
                self.ocr_detection = detection
                return detection
        return None
    
    def get_rgb(self):
        for state in possible_states:
            team_count = self.team_counts[state]
            if not all(value == 0 for value in team_count.values()):
                cluster = max(team_count, key=team_count.get)
                self.team_id = cluster
                return cluster
        return None
    
    def findname(self, player_list):
        for player in player_list:
            shirt_num = int(player["shirt_number"])
            if (self.ocr_detection) == shirt_num:
                self.name = f"{player['name']} ({shirt_num})"
                return self.name
        self.name = None
        return self.ocr_detection
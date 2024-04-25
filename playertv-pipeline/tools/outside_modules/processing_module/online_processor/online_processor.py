import numpy as np
import json
from collections import defaultdict
import copy
import os
import os.path as osp
from loguru import logger
from numpyencoder import NumpyEncoder

from get_metadata.get_metadata import resolve_color
from .online_tracklet import OnlineTracklet
from .danger_zone import DangerZone
from ..processor import Processor

class OnlineProcessor(Processor):
    def __init__(self, params, rgb_module, ocr_module, predictor):
        super().__init__(params, rgb_module, ocr_module, predictor)
        self.online_path = None
        self.final_path = None
        self.set_folders()
        self.params = params
        self.danger_zone_iou_thres = params["params"]["danger_zone_iou_thres"]
        OnlineTracklet.danger_zone_iou_thresh = self.danger_zone_iou_thres
        self.postprocess = params["params"]["post_process"]
        self.final_json["metadata"] = {}
        self.final_json["frame"] = defaultdict(lambda: [])

        self.api_info = None
        if self.get_metadata(self.path):
            self.api_info = self.final_json["metadata"]["api_info"]
            
        self.online_json = copy.deepcopy(self.final_json)
        self.all_tracklets = []

        self.tracklet_mapping = defaultdict(lambda: None)
        self.danger_zones = []


    def set_results_folder(self, folder):
        self.results_folder = folder

    def set_video_path(self, video_path):
        self.video_path = video_path

    def set_paddle_detections(self, mydict):
        self.paddle_detections = mydict

    def rgb_convertion(self, rgb_value):
        rgb_module = self.rgb_module
        return rgb_module.convert_rgb(rgb_value)
    

    def check_danger_zones(self, online_targets, indexes, frame_num, iou_matrix):
        for t in online_targets:
            mytracklet = self.tracklet_mapping[t.track_id]
            if mytracklet == None:
                mytracklet = OnlineTracklet(t.track_id, frame_num)
                self.tracklet_mapping[t.track_id] = mytracklet
                self.all_tracklets.append(mytracklet)
            mytracklet.make_certain()
        for i, t in enumerate(online_targets):
            mytracklet = self.tracklet_mapping[t.track_id]
            iou_sum = self.iou_sum(iou_matrix, i)
            mytracklet.present_in_frame(frame_num)

            if i in indexes[0]:
                sample_idx = np.where(indexes[0]==i)[0]
                sample2 = indexes[1][sample_idx]
                sample2 = sample2[0]
                orig_sample = online_targets[sample2]
                sample2 = self.tracklet_mapping[orig_sample.track_id]
                if sample2 == None:
                    sample2 = OnlineTracklet(orig_sample.track_id, frame_num)
                    self.tracklet_mapping[orig_sample.track_id] = sample2
                mytracklet.make_uncertain()
                sample2.make_uncertain()
                id1, id2 = indexes[0][sample_idx][0], indexes[1][sample_idx][0]
                self.check_danger_zone(mytracklet, sample2, frame_num, iou_matrix[id1, id2])

    def create_dz(self, tracka, trackb, iou, frame_num):
        dz = DangerZone(frame_num, [tracka, trackb], iou)
        tracka.enter_dz(dz)
        trackb.enter_dz(dz)
        self.danger_zones.append(dz)
    
    def resolve_conflict(self, mytracklet, other_tracklet):
        state_dict = {"certain": 3, "uncertain": 2, "danger_zone": 1}
        mytop5, num_ocr = mytracklet.get_ocr_top5()
        othertop5,other_num_ocr = other_tracklet.get_ocr_top5()
        for i in range(mytracklet.taken_index):
            if num_ocr > 0:
                mytop5["detections"].pop(0)
                mytop5["scores"].pop(0)
                mytop5["states"].pop(0)
                num_ocr -= 1

        for i in range(other_tracklet.taken_index):
            if other_num_ocr > 0:
                othertop5["detections"].pop(0)
                othertop5["scores"].pop(0)
                othertop5["states"].pop(0)
                other_num_ocr -= 1

        if other_tracklet.last_frame < mytracklet.last_frame:
            team = int(mytracklet.get_rgb())
            kit_num = int(mytop5["detectoins"][0])
            self.team_kit_numbers[team][kit_num] = mytracklet
            return
        
        mystate = mytop5["states"][0]
        mystate = state_dict[mystate]
        otherstate = othertop5["states"][0]
        otherstate = state_dict[otherstate]

        keep_tracklet = other_tracklet
        send_tracklet = mytracklet
        det = othertop5["detections"]
        det_send = mytop5["detections"]
        if mystate == otherstate:
            myscore = mytop5["scores"][0]
            otherscore = othertop5["scores"][0]
            if myscore > otherscore:
                keep_tracklet = mytracklet
                send_tracklet = other_tracklet
                inter = det_send
                det_send = det
                det = inter
        if mystate > otherstate:
            keep_tracklet = mytracklet
            send_tracklet = other_tracklet
            inter = det_send
            det_send = det
            det = inter

        kit = int(det[0])
        keep_tracklet.set_ocr(kit)
        self.team_kit_numbers[team][kit] = keep_tracklet

        if len(det_send) > 1:
            det_send.pop(0)
            mydet = int(det_send[0])
            third_tracklet = self.team_kit_numbers[team][mydet]
            if third_tracklet == None or third_tracklet == send_tracklet:
                self.team_kit_numbers[team][mydet] = send_tracklet
                return
            else:
                send_tracklet.taken_index += 1
                return self.resolve_conflict(send_tracklet, third_tracklet)
        send_tracklet.set_ocr(None)  
        return

    
    def writeline(self, t, mytracklet, frame_num):
        mytracklet = self.tracklet_mapping[t.track_id]
        track_id = mytracklet.id
        coord = t.last_tlwh
        conf = t.score
        iou = mytracklet.iou
        cluster = mytracklet.get_rgb()
        if cluster == None:
            cluster = -1
        else:
            if self.cluster_map:
                cluster = self.cluster_map[cluster]
        kit_num = mytracklet.ocr_detection
        if kit_num == None:
            kit_num = "NA"
        else:
            kit_num = mytracklet.name

        json_entry = {
            track_id: {
                "frame": frame_num,
                "track_id": track_id,
                "xywh": coord,
                "conf": conf,
                "team_id": cluster,
                "kit_number": kit_num,
            }
        }
        self.online_json["frame"][frame_num].append(json_entry)

    def update_cluster_map(self):
        try:
            cluster_centers = self.rgb_module.cluster_centers
        except Exception as e:
            self.cluster_map = None
            return
        if self.api_info:
            metadata = resolve_color(self.final_json["metadata"]["game_id"], cluster_centers[0], cluster_centers[1], api_info=self.api_info, rgb_module = self.rgb_module)
            
            home_team_idx = int(metadata["home_team_idx"])
            away_team_idx = int(metadata["away_team_idx"])
            home_team = metadata["home_team"]
            away_team = metadata["away_team"]
            self.cluster_map = {home_team_idx: home_team, away_team_idx: away_team}
            self.final_json["metadata"]["api_info"] = metadata

    def my_ocr_score_function(self, txt, conf, iou):
        try:
            txt = int(txt)
            if 0 < txt < 100:
                return conf*(1-iou)
        except:
            return -1.0
        return -1.0
    
    def update_dz(self, tracka, trackb, iou, frame_num):
        create_new = True
        for dz in self.danger_zones:
            in_dz = dz.check_samples(tracka, trackb, frame_num, iou)
            if in_dz:
               create_new = False
               return
            
        if create_new:
            if tracka.is_certain() or trackb.is_certain():
                self.create_dz(tracka, trackb, iou, frame_num)

    def update_states(self, online_targets, frame_num, iou_matrix):
        indexes = np.where(iou_matrix > OnlineTracklet.danger_zone_iou_thresh)
        for i, t in enumerate(online_targets):
            iou_sum = self.iou_sum(iou_matrix, i)
            mytracklet = self.tracklet_mapping[t.track_id]
            if mytracklet == None:
                mytracklet = OnlineTracklet(t.track_id, frame_num, iou_sum)
                self.tracklet_mapping[t.track_id] = mytracklet
                self.all_tracklets.append(mytracklet)

            mytracklet.update_state(iou_sum, frame_num)
    
        for row_index, col_index in zip(*indexes):
            orig_tracklet1 = online_targets[row_index]
            mytracklet1 = self.tracklet_mapping[orig_tracklet1.track_id]
            orig_tracklet2 = online_targets[col_index]
            mytracklet2 = self.tracklet_mapping[orig_tracklet2.track_id]
            iou = iou_matrix[row_index, col_index]

            self.update_dz(mytracklet1, mytracklet2, iou, frame_num)

    def create_results(self, online_targets, frame, frame_num):
        for t in online_targets:
            mytracklet = self.tracklet_mapping[t.track_id]
            loaded_crop = False
            txts = None
            cluster = None
            
            if mytracklet.run_ocr():
                loaded_crop = True
                crop = self.get_crop_from_frame(frame, t.last_tlwh)
                txts, boxes, conf = self.ocr_module.run_ocr(crop)
                if txts != None:
                    for i in range(len(txts)):
                        txt = txts[i]
                        confidence = conf[i]
                        if confidence > OnlineTracklet.conf_thresh:
                            total_score = self.my_ocr_score_function(txt, confidence, mytracklet.iou)
                            team_id = mytracklet.get_team_id()
                            player_list = None
                            if team_id != None:
                                if self.cluster_map:
                                    team_name = self.cluster_map[int(team_id)]
                                    if team_name == self.api_info["home_team"]:
                                        player_list = self.api_info["home_team_player"]
                                    elif team_name == self.api_info["away_team"]:
                                        player_list = self.api_info["away_team_player"]
                                    else:
                                        player_list = None
                                    
                            mytracklet.add_ocr_detection(txt, total_score, player_list)
                            
            if mytracklet.run_rgb():
                if not loaded_crop:
                    crop = self.get_crop_from_frame(frame, t.last_tlwh)
                rgb_val = self.rgb_module.get_avg_rgb(crop)
                cluster = self.rgb_module.predict_cluster(rgb_val)
                if cluster != None:
                    mytracklet.add_rgb(rgb_val, cluster)

            if mytracklet.state == "certain" and mytracklet.post_dz:
                dz = mytracklet.dz
                dz.check_consistency(txts, conf, cluster, mytracklet)

            kit_num, team_id = mytracklet.get_ocr(), mytracklet.get_team_id()
            if kit_num != None and team_id != None:
                team_id, kit_num = int(team_id), int(kit_num)
                other_tracklet = self.team_kit_numbers[team_id][kit_num]
                if other_tracklet != mytracklet and other_tracklet != None:
                    self.resolve_conflict(mytracklet, other_tracklet)


            self.writeline(t, mytracklet, frame_num)
        
        self.rgb_module.update_clusters()
        self.update_cluster_map()

    def post_process(self):
        if self.postprocess:
            self.rgb_module.postprocess_samples()
            cluster_centers = self.rgb_module.cluster_centers
            if self.get_metadata(self.path, cluster_centers[0], cluster_centers[1], cluster=True):
                self.api_info = self.final_json["metadata"]["api_info"]
            post_process_id_map = {}
            for sample in self.all_tracklets:
                if len(sample.rgb_values) > 2:
                    cluster0 = 0
                    cluster1 = 0
                    for value in sample.rgb_values:
                        cluster = self.rgb_module.predict_cluster(value)
                        if cluster == 0:
                            cluster0 += 1
                        elif cluster == 1:
                            cluster1 += 1
                    if cluster0 > cluster1:
                        cluster = 0
                    elif cluster1 > cluster0:
                        cluster = 1
                    else:
                        cluster = None
                if cluster == None:
                    cluster = self.rgb_module.predict_cluster(sample.rgb_mean)

                sample.team_id = cluster
                if self.cluster_map:
                    team_name = self.cluster_map[int(cluster)]
                    if team_name == self.api_info["home_team"]:
                        player_list = self.api_info["home_team_player"]
                    elif team_name == self.api_info["away_team"]:
                        player_list = self.api_info["away_team_player"]
                    else:
                        player_list = None


                    name = sample.findname(player_list)
                    entry = {"team":team_name, "kit_number": name}
                    post_process_id_map[sample.id] = entry

        frames = self.online_json["frame"]
        for frame, samples in frames.items():
            for sample in samples:
                for track_id, info in sample.items():
                    mytracklet = self.tracklet_mapping[int(track_id)]
                    if self.cluster_map:
                        team = self.cluster_map[mytracklet.team_id]
                    else:
                        team = mytracklet.team_id

                    entry = {mytracklet.id: {
                        "frame": frame,
                        "track_id": mytracklet.id,
                        "xywh": info["xywh"],
                        "conf": info["conf"],
                        "team_id": team,
                        "kit_number": f"{mytracklet.name}" if mytracklet.name else "NA"
                    }}
                    self.final_json["frame"][frame].append(entry)

        self.save_results()


    def process_frame(self, online_targets, frame, frame_num, iou_matrix):
        self.rgb_module.new_frame()
        self.update_states(online_targets, frame_num, iou_matrix)
        self.create_results(online_targets, frame, frame_num)
        self.write_online()

    def set_folders(self):
        path = self.path
        path = osp.basename(self.path)
        path = osp.join("results", path.split(".")[0])
        os.makedirs(path, exist_ok=True)

        self.online_path = osp.join(path, "online_results.json")
        logger.info(f"Online output path: '{self.online_path}'")
        self.final_path = osp.join(path, "final_results.json")

    def write_online(self):
        with open(self.online_path, "w", encoding="utf-8") as file:
            json_data = json.dumps(self.online_json, indent=4, ensure_ascii=False, cls=NumpyEncoder)
            file.write(json_data)
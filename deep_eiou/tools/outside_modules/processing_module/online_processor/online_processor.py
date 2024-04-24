import numpy as np
import json
from collections import defaultdict
from typing import Union

from ..post_processer.helper_functions.helper_functions import get_crop_from_frame

from .online_tracklet import OnlineTracklet
from .danger_zone import DangerZone
from ..post_processer.smart_crop_extractor import Sample_Crop
from .danger_zone import IdSwitch, KitNumberClash
from ..processor import Processor

class OnlineProcessor(Processor):
    def __init__(self, params, rgb_module, ocr_module, predictor):
        super().__init__(params, rgb_module, ocr_module, predictor)
        self.params = params
        self.time_thres = params["params"]["time_thres"]
        self.danger_zone_iou_thres = params["params"]["danger_zone_iou_thres"]
        self.all_tracklets = []
        self.preprocessed_txt = []
        self.postprocessed_txt = []

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
    
    ##def set_metadata(self, metadata):
     #   self.metadata = metadata
     #   home_team_idx = int(metadata["home_team_idx"])
     #   away_team_idx = int(metadata["away_team_idx"])
     #   home_team = metadata["home_team"]
      #  away_team = metadata["away_team"]
      #  self.cluster_map = {home_team_idx: home_team, away_team_idx: away_team}

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
                #print(sample2)
                #print(sample_idx)
                mytracklet.make_uncertain()
                sample2.make_uncertain()
                id1, id2 = indexes[0][sample_idx][0], indexes[1][sample_idx][0]
                print("DANGERZONE", id1, id2)
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
        rgb = -1
        brisque = -1
        iou = mytracklet.iou
        cluster = mytracklet.get_rgb()
        if cluster == None:
            cluster = -1
        else:
            if self.cluster_map:
                cluster = self.cluster_map[cluster]
        kit_num = mytracklet.ocr_detection
        if kit_num == None:
            kit_num = -1


        txt = f"{frame_num}, {track_id}, {coord[0]}, {coord[1]}, {coord[2]}, {coord[3]}, {conf}, {rgb}, {brisque}, {iou}, {cluster}, {kit_num}\n"
        self.preprocessed_txt.append(txt)
        json_entry = {
            track_id: {
                "frame": frame_num,
                "track_id": track_id,
                "xywh": coord,
                "conf": conf,
                "avg_rgb": rgb,
                "brisque_score": brisque,
                "iou": iou,
                "team": cluster,
                "player": kit_num
            }
        }
        self.final_json["frame"][frame_num].append(json_entry)

    def ocr_score_function(self, txt, conf, iou):
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
                print("Danger_zone_created")
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
            if self.paddle_detections:
                try:
                    txts, boxes, conf = self.paddle_detections[str(frame_num)][str(mytracklet.id)]
                except:
                    txts, boxes, conf = None, None, None

                if txts != None:
                    for i in range(len(txts)):
                        txt = txts[i]
                        confidence = conf[i]
                        if confidence > 0.5:
                            total_score = self.ocr_score_function(txt, confidence, mytracklet.iou)
                            team_id = mytracklet.get_team_id()
                            player_list = None
                            if team_id:
                                if self.cluster_map:
                                    team_name = self.cluster_map[int(team_id)]
                                    if team_name == self.metadata["home_team"]:
                                        player_list = self.metadata["home_team_player"]
                                    elif team_name == self.metadata["away_team"]:
                                        player_list = self.metadata["away_team_player"]
                                    else:
                                        player_list = None
                            mytracklet.add_ocr_detection(txt, total_score, player_list)
            elif mytracklet.run_ocr():
                loaded_crop = True
                crop = get_crop_from_frame(frame, t.last_tlwh)
                txts, boxes, conf = self.ocr_module.run_ocr(crop)
                self.all_ocr_detections[frame_num][mytracklet.id] = (txts, boxes, conf)
                if txts != None:
                    for i in range(len(txts)):
                        txt = txts[i]
                        confidence = conf[i]
                        if confidence > 0.5:
                            total_score = self.ocr_score_function(txt, confidence, mytracklet.iou)
                            team_id = mytracklet.get_team_id()
                            player_list = None
                            if team_id:
                                if self.cluster_map:
                                    team_name = self.cluster_map[int(team_id)]
                                    if team_name == self.metadata["home_team"]:
                                        player_list = self.metadata["home_team_player"]
                                    elif team_name == self.metadata["away_team"]:
                                        player_list = self.metadata["away_team_player"]
                                    else:
                                        player_list = None
                            mytracklet.add_ocr_detection(txt, total_score, player_list)
                            
            if mytracklet.run_rgb():
                if not loaded_crop:
                    crop = get_crop_from_frame(frame, t.last_tlwh)
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

    def post_process(self):
        self.rgb_module.postprocess_samples()
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
            if self.cluster_map:
                team = self.cluster_map[int(cluster)]
                entry = {"team":team, "kit_number": sample.ocr_detection}
                post_process_id_map[sample.id] = entry
        return post_process_id_map


    def process_frame(self, online_targets, frame, frame_num, iou_matrix):
        self.rgb_module.new_frame()
        self.update_states(online_targets, frame_num, iou_matrix)
        self.create_results(online_targets, frame, frame_num)














"""
##########################################################################################################
        for i, t in enumerate(online_targets):
            crop_loaded = False
            mytracklet = self.tracklet_mapping[t.track_id]
            if mytracklet.uncertain:
                self.writeline(t, mytracklet, frame_num)
                continue
            
            iou_sum = self.iou_sum(iou_matrix, i)
            mytracklet.set_iou(iou_sum)
            rgb_val = None
            if iou_sum < self.rgb_module.iou_thresh or mytracklet.get_team_id() == None:
                if not crop_loaded:
                    crop = get_crop_from_frame(frame, t.last_tlwh)
                rgb_val = self.rgb_module.get_avg_rgb(crop)

                ## See if we have online fitted alreay, otherwise wait
                cluster = self.rgb_module.predict_cluster(rgb_val)
                if cluster != None:
                    mytracklet.add_rgb(rgb_val, cluster, iou_sum)

            if mytracklet.get_ocr() == None or 1==1:
                crop_loaded = True
                crop = get_crop_from_frame(frame, t.last_tlwh)
                txts, boxes, conf = self.ocr_module.run_ocr(crop)
                if txts != None:
                    for i in range(len(txts)):
                        txt = txts[i]
                        confidence = conf[i]
                        total_score = self.ocr_score_function(txt, confidence, iou_sum)
                        mytracklet.add_ocr_detection(txt, total_score)
                mytracklet.check_dz(txts, conf, rgb_val, iou_sum)


            team, kit = mytracklet.get_team_id(), mytracklet.get_ocr()
            if team != None and kit != None:
                sample2 = self.team_kit_numbers[team][kit]
                if sample2 == None or sample2 == mytracklet:
                    #print("Found Place")
                    self.team_kit_numbers[team][kit] = mytracklet
                else:
                    print("clash")
                    #print(sample2.ocr_detections, sample2)
                    #print(mytracklet.ocr_detections, mytracklet)
                    #self.resolve_position(mytracklet, sample2)


            #self.create_dz(online_targets, indexes)

            self.writeline(t, mytracklet, frame_num)
        self.rgb_module.update_clusters()"""
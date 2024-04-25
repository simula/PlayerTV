import json
import os.path as osp
import numpy as np
import cv2

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from collections import defaultdict

from tools.outside_modules.rgb_module.distance_metrics_rgb import weighted_rgb_metric

from post_processing.smart_crop_detector import Sample_Crop

def split_into_equal_parts(arr, num_parts):
    avg = len(arr)/float(num_parts)
    last = 0.0
    out = []
    while last < len(arr):
        out.append(arr[int(last): int(last + avg)])
        last += avg
    return out

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

def get_frame_by_number(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        return frame
    else:
        return None
    

def get_ocr_final_results(ocr_results):
    assert isinstance(ocr_results, (dict, str)), "Not dictionary or string"
    #Assuming json file string
    if isinstance(ocr_results, str):
        with open(ocr_results, "r") as file:
            ocr_results = json.load(file)
    
    final_results = {}
    for track_id, info in ocr_results.items():
        top_score = 0
        final_txt = "NA"
        accumulator_dict = defaultdict(int)
        for i in range(len(info["txt"])):
            for j in range(len(info["txt"][i])):
                txt = info["txt"][i][j]
                conf = info["conf"][i][j]
                try:
                    iou = info["iou"][i]
                except:
                    print(info)
                    exit()
                accumulator_dict[txt] += 1
                try:
                    num_size = len(txt)
                    if num_size > 2:
                        num_size = 2
                    score = conf * accumulator_dict[txt] * iou
                    txt = int(txt)
                    if 0 < txt < 100:
                        score = score *2
                except:
                    score = 0

                if score > top_score:
                    top_score = score
                    final_txt = txt

        final_results[int(track_id)] = final_txt
    return final_results


def get_rgb_clusters(avg_rgb_list):
    k = 2
    normalized_rgb = normalize(avg_rgb_list.astype(float), axis=1, norm="l2")
    #normalized_rgb = normalized_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(normalized_rgb)
    normalized_cluster_means = kmeans.cluster_centers_
    print(normalized_cluster_means)
    cluster_means = normalized_cluster_means * 255

    rgb_info = {"normalized_cluster_means": normalized_cluster_means.tolist(),
                "cluster_means": cluster_means.tolist()}
    
    return cluster_labels, rgb_info

def get_team_mapping(rgb_module, rgb_cluster_means, players_dict, api_map):
    avg_rgb_list = rgb_module.samples
    rgb_cluster_labels = rgb_module.get_cluster_labels()
    home_team_idx = api_map["home_team_idx"]
    away_team_idx = api_map["away_team_idx"]
    team_map = {}

    for id, samples in players_dict.items():
        home = 0
        away = 0
        for sample in samples:
            rgb = sample["avg_rgb"]
            team_id = np.where(avg_rgb_list == rgb)[0]

            if len(team_id) > 0:
                ind = team_id[0]
                team_id = rgb_cluster_labels[ind]
            
            else:
                distance = 1000000
                idx = None
                for i in range(len(rgb_cluster_means)):
                    dist = weighted_rgb_metric(rgb, rgb_cluster_means[i])
                    if dist < distance:
                        team_id = i
                        distance = dist
        
            if team_id == home_team_idx:
                home += 1
            else:
                away += 1

        if home >= away:
            team_id = api_map["home_team"]
            idx = home_team_idx
        else:
            team_id = api_map["away_team"]
            idx = away_team_idx
        
        team_map[int(id)] = {"team_name": team_id, "idx": idx}

    return team_map

def create_line_team_ocr(txt_line, rgb_module, ocr_final, final_json, player_team_mapping, team_mapping=None):
    avg_rgb_list = rgb_module.samples
    clusters = rgb_module.get_cluster_labels
    sample = Sample_Crop(txt_line)
    home_team_idx = player_team_mapping["home_team_idx"]
    if team_mapping is not None:
        team_id = team_mapping[sample.id]["idx"]

    else:
        team_id = np.where(avg_rgb_list == sample.avg_rgb)[0]
        if len(team_id) > 0:
            ind = team_id[0]
            team_id = clusters[ind]
        else:
            AssertionError(f"Did not find correct avg rgb\nDid not find value {sample.avg_rgb}")

    if team_id == home_team_idx:
        team_id = player_team_mapping["home_team"]
        players_list = player_team_mapping["home_team_player"]
    else:
        team_id = player_team_mapping["away_team"]
        players_list = player_team_mapping["away_team_player"]
        
    sample.set_team(team_id)
    kit_num_str = decide_kit_num(sample, ocr_final, players_list)
                
    sample.set_ocr(kit_num_str)
    
    if final_json:
        final_json["frame"][sample.frame][sample.id]["team_id"] = sample.team
        final_json["frame"][sample.frame][sample.id]["kit_number"] = sample.ocr


    strline = f"{sample.frame}, {sample.id}, {sample.xywh[0]}, {sample.xywh[1]}, {sample.xywh[2]}, {sample.xywh[3]}, {sample.conf}, {sample.avg_rgb}, {sample.brisque}, {sample.iou}, {sample.team}, {sample.ocr}"
    return strline


def decide_kit_num(sample, ocr_results, players_list):
    track_id = sample.id
    top_score = 0
    final_txt = "NA"
    accumulator_dict = defaultdict(int)
    try:
        info = ocr_results[str(track_id)]
        for i in range(len(info["txt"])):
            for j in range(len(info["txt"][i])):
                txt = info["txt"][i][j]
                conf = info["conf"][i][j]
                iou = info["iou"][i]
                accumulator_dict[txt] += 1
                try:
                    num_size = len(txt)
                    if num_size > 2:
                        num_size = 2
                    score = conf * accumulator_dict[txt] * (1-iou)
                    kit_num = int(txt)
                    for player in players_list:
                        name = player["name"]
                        player_kit = int(player["shirt_number"])
                        if player_kit == kit_num:
                            score = score * 10
                            txt = f"{name} ({txt})"

                except:
                    score = 0

                if score > top_score:
                    top_score = score
                    final_txt = txt
    except:
        final_txt = "NA"

    return final_txt
import json
import os.path as osp
import os
import cv2

from collections import defaultdict
from numpyencoder import NumpyEncoder

from tools.outside_modules.ocr_module.ocr_module import OCR_Module, PaddleOCRModule, EasyOCRModule

from post_processing.helper_functions.helper_functions import get_crop_from_frame, get_frame_by_number, split_into_equal_parts
from post_processing.smart_crop_detector import Smart_Crop_Detector, Sample_Crop

with open("config.json", "r") as file:
        configs = json.load(file)

def get_smart_ocr_results(video_path, results, ocr_module = None, out_file_folder="test/smart_crop"):
    smart_crop_detector = Smart_Crop_Detector()
    if not ocr_module:
        ocr_config = configs["ocr_parameters"]
        if ocr_config["paddle"]:
            ocr_module = PaddleOCRModule(ocr_config["paddle_params"])
        elif ocr_config["easyocr"]:
            ocr_module = EasyOCRModule(ocr_config["easyocr_params"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file. Do you have the correct path?\nPath provided: {video_path}")
    
    assert isinstance(results, (list, str)), "Not list or string"
    #Assuming txt file string
    if isinstance(results, str):
        with open(results, "r") as file:
            for line in file:
                sample = Sample_Crop(line)
                smart_crop_detector.check_sample(sample)
    #Assuming list
    else:
        for line in results:
            sample = Sample_Crop(line)
            smart_crop_detector.check_sample(sample)

    #This is a defaultdict with id at key and a list of samples sorted in ascending order by score
    good_samples = smart_crop_detector.return_samples()

    ocr_results = defaultdict(lambda: {"frame": [], "txt": [], "conf": [], "boxes": [], "iou": []})
    for _, sample_list in good_samples.items():
        for sample in sample_list:
            if sample is not None:
                print("Running sample")
                frame_number = sample.frame
                frame = get_frame_by_number(cap, frame_number)
                crop = get_crop_from_frame(frame, sample.xywh)

                txts, boxes, scores = ocr_module.run_ocr(crop)
                if txts:
                    ocr_results[int(sample.id)]["frame"].append(frame_number)
                    ocr_results[int(sample.id)]["txt"].append(txts)
                    ocr_results[int(sample.id)]["conf"].append(scores)
                    ocr_results[int(sample.id)]["boxes"].append(boxes)
                    ocr_results[int(sample.id)]["iou"].append(sample.iou)
                    for txt in txts:
                        try:
                            txt = int(txt)
                            print("Found number", txt)
                            
                        except:
                            print("found txt:,", txt)

    os.makedirs(out_file_folder, exist_ok=True)
    out_path = osp.join(out_file_folder, "ocr_results.json")
    ocr_results = dict(ocr_results)
    json_ocr_results = json.dumps(ocr_results, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    with open(out_path, "w") as file:
        file.write(json_ocr_results)

    return ocr_results


def get_smarter_ocr_results(video_path, player_dict, ocr_module = None, out_file_folder="test/smart_crop"):
    smart_crop_detector = Smart_Crop_Detector()
    if not ocr_module:
        ocr_config = configs["ocr_parameters"]
        if ocr_config["which_ocr"] == "paddle":
            ocr_module = PaddleOCRModule(ocr_config["paddle_params"])
        elif ocr_config["which_ocr"] == "easyocr":
            ocr_module = EasyOCRModule(ocr_config["easyocr_params"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file. Do you have the correct path?\nPath provided: {video_path}")
    
    ocr_results = defaultdict(lambda: {"frame": [], "txt": [], "conf": [], "boxes": [], "iou": []})
    for tid, frames in player_dict.items():
        split_list = split_into_equal_parts(frames, 5)
        top_top_samples = []
        for time_slot in split_list:
            for frame in time_slot:
                smart_crop_detector.check_sample_player(frame)

            good_samples = smart_crop_detector.return_samples()
            top_top_samples.append(good_samples)
            smart_crop_detector.reset_samples()
            
        for time_slots in top_top_samples:
            found = False
            for id, samples in time_slots.items():
                for sample in samples:
                    if sample is not None:
                        frame_number = sample["frame"]
                        frame = get_frame_by_number(cap, frame_number)
                        crop = get_crop_from_frame(frame, sample["xywh"])
                        txts, boxes, scores = ocr_module.run_ocr(crop)
                        print("track_id", sample["track_id"])
                        if txts:
                            ocr_results[int(sample["track_id"])]["frame"].append(frame_number)
                            ocr_results[int(sample["track_id"])]["txt"].append(txts)
                            ocr_results[int(sample["track_id"])]["conf"].append(scores)
                            ocr_results[int(sample["track_id"])]["boxes"].append(boxes)
                            ocr_results[int(sample["track_id"])]["iou"].append(sample["iou"])
                            for txt in txts:
                                try:
                                    txt = int(txt)
                                    print("Skipping time period:", txt)
                                    found = True
                                    
                                except:
                                    print("found txt:,", txt)
                                   
                    if found:
                        break
    os.makedirs(out_file_folder, exist_ok=True)
    out_path = osp.join(out_file_folder, "ocr_results.json")
    ocr_results = dict(ocr_results)
    json_ocr_results = json.dumps(ocr_results, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    with open(out_path, "w") as file:
        file.write(json_ocr_results)

    return ocr_results
import cv2
import json
import torch
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import os.path as osp
import re
import os

from loguru import logger
from collections import defaultdict
from tools.argparser.argparser import create_args
from tools.data_augmentation.data_augmentation import preproc

from tools.tracker.Deep_EIoU import Deep_EIoU
from tools.utils.timer import Timer
from tools.exp.exp import Exp
from tools.feature_extractor import FeatureExtractor

from get_metadata.get_metadata import resolve_color
from helper_functions import save_file, save_numpy_array, return_capture_metadata, create_results_folder, get_crop_from_frame, get_modules
from tools.outside_modules.processing_module.online_processor.online_processor import OnlineProcessor


def run(predictor, extractor, args, processor):
    cap, metadata = return_capture_metadata(args.path)
    processor.cap = cap

    processor.set_metadata(metadata)
    width = metadata["width"]
    height = metadata["height"]
    fps = metadata["fps"]

    tracker = Deep_EIoU(args, frame_rate=fps)
    timer = Timer()

    frame_id = 0

    while True:
        if frame_id % 30 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        if args.frame_skip > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret_val, frame = cap.read()
        if ret_val:
            timer.tic()
            online_targets = []
            if processor.online_target_loader != None:
                online_targets, iou_matrix = processor.online_target_loader()
                
            else:
                outputs, img_info = predictor.inference(frame, timer)
                # outputs is list of results. idx 0 is results idx 1 is device
                if outputs[0] is None:
                    det = None
                else:
                    det = outputs[0].cpu().detach().numpy()
                    scale = min(1440/width, 800/height)
                    det /= scale
                    cropped_imgs = [frame[max(0,int(y1)):min(height,int(y2)),max(0,int(x1)):min(width,int(x2))] for x1,y1,x2,y2,_,_,_ in det]

                    ## Cropped images contains all the crops in BGR format
                    embs = extractor(cropped_imgs)

                if det is not None:
                    embs = embs.cpu().detach().numpy()
                    ## Only return online targets, processor will do the iou matrix creating
                    online_targets, iou_matrix = tracker.update(det, embs, args.min_box_area)
                
            if len(online_targets) == 0:
                frame_id += args.frame_skip
                timer.toc()
                continue
            np.fill_diagonal(iou_matrix, 0)
            iou_matrix = np.triu(iou_matrix)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processor.process_frame(online_targets, frame_rgb, frame_id, iou_matrix)
            timer.toc()
                                
            if args.save_video:
                pass
                ## TODO: IMPLEMENT OWN visualizer for saving video
                ## video_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += args.frame_skip

    return processor

def main(args, processor):
    logger.info(f"Args: {args}")
    predictor = processor.predictor
    extractor = FeatureExtractor(model_name='osnet_x1_0', model_path='checkpoints/sports_model.pth.tar-60', device=args.device)

    processor = run(predictor, extractor, args, processor)
    return processor

def run_deep_eiou(video_path=None):
    with open("config.json", "r") as file:
        configs = json.load(file)

    deep_eiou_parameters = configs["deep_eiou_parameters"]
    if video_path:
        deep_eiou_parameters["video_path"] = video_path
        configs["processor_parameters"]["path"] = deep_eiou_parameters["video_path"]
    else:
        video_path = deep_eiou_parameters["video_path"]
        configs["processor_parameters"]["path"] = deep_eiou_parameters["video_path"]

    args = create_args(deep_eiou_parameters, "Deep_EIoU_parameters")
    args.device = "cuda" if args.device == "gpu" else "cpu"
    args.path = deep_eiou_parameters["video_path"]
    #args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
    processor = get_modules(configs, args)


    regex_clip_match = r"((\d{4})_(.*?))\."
    match = re.search(regex_clip_match, video_path)
    if match:
        basename = match.group(1)
        cached_results_folder = osp.join("cached_results", basename)

        predictor_type = configs["predictor_params"]["type"]
        cached_track = osp.join(cached_results_folder, predictor_type, "tracker_results.txt")
        if osp.exists(cached_track):
            processor.load_tracker_results(cached_ocr)
        
        ocr_type = configs["ocr_parameters"]["which_ocr"]
        cached_ocr = osp.join(cached_results_folder, ocr_type, "all_detections.json")
        if osp.exists(cached_ocr):
            processor.load_ocr_detections(cached_ocr)


    processor = main(args, processor)
    processor.post_process()
    processor.cap.release()
    return processor




if __name__ == "__main__":
    run_deep_eiou()

    

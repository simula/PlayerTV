import cv2
import json
import datetime
import numpy as np
import os.path as osp
import os
import torch
import re

from loguru import logger
from typing import Union
from numpyencoder import NumpyEncoder

from get_metadata.get_metadata import get_video_url

from image_folder_capture import ImageFolderCapture
from tools.predictor import Predictor
from tools.exp.get_exp import get_exp
from tools.utils.utils import get_model_info

from tools.outside_modules.predictor.yolov8_predictor.yolov8_predictor import Yolov8Predictor
from tools.outside_modules.rgb_module.rgb_module import RGBModule, DBSCANModule, KMeansModule
from tools.outside_modules.rgb_module.online.online_rgb_module import OnlineRGBModule
from tools.outside_modules.ocr_module.ocr_module import OCR_Module, PaddleOCRModule, EasyOCRModule
from tools.outside_modules.processing_module.post_processer.post_processor import PostProcessor
from tools.outside_modules.processing_module.online_processor.online_processor import OnlineProcessor

regex_clip_match = r"(\d{4})_(.*?)\."

def save_file(content: Union[list, dict], folder: str, filename: str, format: str, comment: str = None):
    assert isinstance(content, (list, dict))
    os.makedirs(folder, exist_ok=True)
    out_file = osp.join(folder, filename)
    out_file = f"{out_file}.{format}"
    with open(out_file, "w", encoding="utf-8") as file:
        if format == "txt":
            file.writelines(content)

        if format == "json":
            json_data = json.dumps(content, indent=4, ensure_ascii=False, cls=NumpyEncoder)
            file.write(json_data)

    if not comment:
        logger.info(f"Saved {format} file to {out_file}")
    else:
        logger.info(f"Saved {comment} to {out_file}")

def save_numpy_array(array: np.array, folder: str, filename: str, comment: str = None):
    folder = osp.join(folder, "numpy")
    os.makedirs(folder, exist_ok=True)
    folder = osp.join(folder, f"{filename}.npy")
    np.save(folder, array)
    if not comment:
        logger.info(f"Saved numpy file file to {folder}")
    else:
        logger.info(f"Saved numpy ({comment}) to {folder}")

def return_capture_metadata(video_path: str):
    if osp.basename(video_path) == "img1":
        cap = ImageFolderCapture(video_path)
        width = 1920
        height = 1080
        fps = 25
        game_id = None
        clip_id = None
        video_url=None
    else:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
        fps = cap.get(cv2.CAP_PROP_FPS)


        match = re.search(regex_clip_match, video_path)
        if match:
            game_id = match.group(1)
            clip_id = match.group(2)
            video_url = get_video_url(clip_id)
        else:
            game_id=None
            clip_id=None
            video_url=None

    metadata = {
        "id": f"{game_id}_{clip_id}" if game_id else None,
        "game_id": game_id,
        "clip_id": clip_id,
        "width": width,
        "height": height,
        "fps": fps,
        "video_url": video_url
    }
    return cap, metadata
    

def create_results_folder(video_path: str, predictor):
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    basename = osp.basename(video_path).split(".")[0]
    save_folder = osp.join("results", basename, predictor.save_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    logger.info(f"Save path is {save_folder}")
    return save_folder

def create_default_predictor(default_params, args):
    exp_file = default_params["exp_file"]
    exp = get_exp(exp_file)
    if args.nms_thresh is not None:
        exp.nmsthre = args.nms_thresh

    model = exp.get_model().to(torch.device(args.device))
    model_summary = get_model_info(model, exp.test_size)
    logger.info(f"Model summary: {model_summary}")
    model.eval()
    ckpt = default_params["ckpt"]

    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done")

    predictor = Predictor(model, exp, args.device, params=default_params)
    return predictor


def get_modules(configs: dict, args):
    ## Predictor
    predictor_params = configs["predictor_params"]
    device = args.device
    if predictor_params["type"] == "yolov8":
        predictor_params["params"] = predictor_params["params"]["yolov8_params"]
        configs["deep_eiou_parameters"]["track_high_thresh"] = predictor_params["params"]["conf_thres"]
        args.track_high_thresh = predictor_params["params"]["conf_thres"]
        configs["deep_eiou_parameters"]["new_track_thresh"] = predictor_params["params"]["conf_thres"]
        args.new_track_thresh = predictor_params["params"]["conf_thres"]
        configs["deep_eiou_parameters"]["nms_thresh"] = predictor_params["params"]["iou_nms_thres"]
        args.nms_thresh = predictor_params["params"]["iou_nms_thres"]
        predictor = Yolov8Predictor(predictor_params["params"], device)

    else:
        default_params = predictor_params["params"]["default_params"]
        predictor_params["params"] = default_params
        exp_file = default_params["exp_file"]
        args.ckpt = exp_file

        predictor = create_default_predictor(default_params, args)


    ## OCR module
    ocr_parameters = configs["ocr_parameters"]
    ocr_type = ocr_parameters["which_ocr"]
    if ocr_type == "paddle":
        params = ocr_parameters["params"]["paddle_params"]
        ocr_parameters["params"] = params
        ocr_module = PaddleOCRModule(params)

    elif ocr_type == "easyocr":
        params = ocr_parameters["params"]["easyocr_params"]
        ocr_parameters["params"] = params
        ocr_module = EasyOCRModule(params)
    else:
        raise NotImplementedError(f"Does not support ocr method: {ocr_type}, check your config file.")
    
    ### RGB module
    rgb_parameters = configs["rgb_parameters"]
    cluster_method = rgb_parameters["cluster_method"]
    offset_type = rgb_parameters["offset_type"]

    # Initializing rgb module
    if offset_type:
        if offset_type == "center":
            rgb_parameters["offset"] = rgb_parameters["offset"]["offset_center"]
        elif offset_type == "trim":
            rgb_parameters["offset"] = rgb_parameters["offset"]["offset_trim"]
        else:
            raise NotImplementedError(f"Does not support offset type: {offset_type}, check your config file.")
    else:
        rgb_parameters["offset"] = None

    if cluster_method == "DBSCAN":
        rgb_module = DBSCANModule(rgb_parameters)
    elif cluster_method == "KMeans":
        rgb_module = KMeansModule(rgb_parameters)
    else:
        NotImplementedError(f"Does not support rgb cluster method: {cluster_method}, check your config file.")


    ##processor module
    processor_parameters = configs["processor_parameters"]
    processor_type = processor_parameters["type"]
    if processor_type == "post":
        processor_parameters["params"] = processor_parameters["params"]["post"]
        processor = PostProcessor(processor_parameters, rgb_module, ocr_module, predictor)
    elif processor_type == "online":
        processor_parameters["params"] = processor_parameters["params"]["online"]
        rgb_module = OnlineRGBModule(rgb_parameters)
        processor = OnlineProcessor(processor_parameters, rgb_module, ocr_module, predictor)
    else:
        raise NotImplementedError(f"Processor type: {processor_type} is not implemented. Check your config file")


    return processor

def get_crop_from_frame(frame, tlwh):
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
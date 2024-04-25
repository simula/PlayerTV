# Configuration File Documentation

## Running the Pipeline

To run this pipeline, set the MP4 video path in the `config.json` file before running the Python script `run_playertv.py`. Ensure that the `video_path` field under `deep_eiou_parameters` is correctly set to point to your video file.
Execution of the pipeline is done by `python run_playertv.py`

## Predictor Parameters

In the `predictor_params` section, you can choose between two types of models: YOLOv8 or the default YOLOX model. This is specified with the `"type"` parameter.

- **Type**: Set to `"default"` for YOLOX or `"yolov8"` for YOLOv8. This choice determines which additional parameters are relevant in the `params` sub-dictionary.
- **YOLOv8 Parameters**: If `type` is `"yolov8"`, configure the model with checkpoints, confidence thresholds, and image size specific to YOLOv8 needs.
- **Default Parameters**: For YOLOX or other default models, specify checkpoints and experimental files necessary for operation.

## OCR Parameters

The `ocr_parameters` section allows you to choose between two OCR engines: EasyOCR and PaddleOCR, controlled by the `which_ocr` parameter.

- **Which OCR**: Set to `"easyocr"` for using EasyOCR and `"paddle"` for using PaddleOCR.
- **Parameters**:
  - If `which_ocr` is `"easyocr"`, set languages and GPU usage.
  - If `which_ocr` is `"paddle"`, configure detailed OCR settings including angle classification, language, GPU usage, and detection thresholds.

## RGB Parameters

This section configures the RGB extraction and analysis settings, particularly useful in clustering and scoring methods.

- **Certain IOU Threshold**: Defines the IOU threshold used as a baseline for clustering.
- **Color Space**: Choose between `"weighted_rgb"`, `"rgb"`, or `"CIELAB"` to represent the color space in the analysis.
- **Normalize** and **Scale Values**: Booleans that control whether the RGB values are normalized and/or scaled.
- **Offset Type**: Determines how the area from the crop is sampled. Options are `"center"`, `"trim"`, or `null`.

## Processor Parameters

Control how information is processed, either in real-time or post-event, along with how detections are scored and team scores are decided.

- **Type**: Choose `"online"` for real-time processing or `"post"` for post-processing. This affects which parameters in `params` are relevant.
- **Stop at Detection**: Boolean to decide whether to stop OCR once a detection is found.
- **Crop Score Function**: Determines the scoring function used to rank crops.
- **OCR Score Function**: Defines how OCR detections are scored.
- **Decide Team Scores**: Method to decide the team for a player, either `"plurality_vote"` or `"min_total_distance"`.

For a full understanding of the configuration options and to ensure correct settings, refer to the individual descriptions of parameters and the accompanying software documentation.

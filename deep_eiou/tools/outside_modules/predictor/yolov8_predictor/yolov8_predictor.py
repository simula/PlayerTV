import torch


from ultralytics import YOLO
from ..my_predictor import MyPredictor
from ..processing import preprocess

class Yolov8Predictor(MyPredictor):

    def __init__(self, yolo_params, device):
        super().__init__(device)
        self.save_folder = "yolov8"
        self.input_size = tuple(yolo_params["inf_img_size"])
        self.params = yolo_params
        self.model = YOLO(yolo_params["ckpt"], verbose=False)

    def get_params(self):
        return self.params

    def standardize_output(self, outputs):
        output = outputs[0].boxes.xyxy.to("cpu")
        conf = outputs[0].boxes.conf.to("cpu")

        if output.size(0) == 0:
            return [None, 0]
        conf = conf.unsqueeze(1)
        classes = outputs[0].boxes.cls.to("cpu").unsqueeze(1)

        result = torch.cat((output, conf, conf, classes), dim=1)
        result = [result, 0]
        return result

    def inference(self, img, timer):
        timer.tic()
        img_info = {"id": 0}

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img


        img, ratio = preprocess(img, self.input_size, self.rgb_means, self.std)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        outputs = self.model(img, stream=False, conf=self.params["conf_thres"], iou=self.params["iou_nms_thres"], classes=self.params["class_whitelist"], verbose=False)

        outputs = self.standardize_output(outputs)
        return outputs, img_info


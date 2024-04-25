import torch

class MyPredictor:
    def __init__(self, device=torch.device("cuda")):
        self.device = device
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        raise NotImplementedError("Implementation is needed in submodules")
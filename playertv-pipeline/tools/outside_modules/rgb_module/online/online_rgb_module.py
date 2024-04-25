import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import normalize

from skimage import color

from ..rgb_module import RGBModule

from tools.outside_modules.rgb_module.get_crop_offset import get_crop_center_offset, get_crop_trim_offset, parse_offset


clustering_methods = {"KMeans": MiniBatchKMeans, "DENSTREAM": None}
get_crop_offset = {"center": get_crop_center_offset, "trim": get_crop_trim_offset}


class OnlineRGBModule(RGBModule):
    def __init__(self, parameters: dict):
        super().__init__(parameters)
        self.KMeans = MiniBatchKMeans(2, batch_size=15)
        self.parameters = parameters


        self.current_values = []

    @property
    def cluster_centers(self):
        return self.KMeans.cluster_centers_

    def new_frame(self):
        self.current_values = []

    def add_current_value(self, value):
        self.current_values.append(value)

    def add_value(self, value: np.array):
        self.add_current_value(value)
        self.all_samples.append(value)
        return True

    def get_avg_rgb(self, crop):
        #score = self.score_function(crop)
        crop = self.get_crop(crop, self.offset)
        rgb_val = self.rgb_function(crop)
        self.add_value(rgb_val)
        return rgb_val
    
    def update_clusters(self):
        new_points = self.current_values
        if len(new_points) < 2:
            return
        if self.scale_values:
            new_points = self.scaler.fit_transform(self.current_values)
        self.KMeans.partial_fit(new_points)

    def convert_rgb(self, rgb_value):
        if self.parameters["color_space"] == "CIELAB":
            rgb_value = color.rgb2lab(rgb_value)
        new_point = rgb_value.reshape(1, -1)
        new_point = self.process_sample(new_point)
        return new_point

    def predict_cluster(self, point):
        try:
            new_point = point
            new_point = new_point.reshape(1, -1)
            new_point = self.process_sample(new_point)
            cluster = self.KMeans.predict(new_point)
            return cluster[0]
        except Exception as e:
            return None

    def get_cluster_center_rgb(self):
        try:
            original_cluster_centers = self.KMeans.cluster_centers_
            if self.scale_values:
                original_cluster_centers = self.scaler.inverse_transform(original_cluster_centers)
            if self.normalize:
                new_point = new_point*(0.59*255)
                
            if self.parameters["color_space"] == "RGB":
                original_cluster_centers = original_cluster_centers/np.array([0.3, 0.59, 0.11])
            elif self.parameters["color_space"] == "CIELAB":
                original_cluster_centers = color.lab2rgb(original_cluster_centers)
            return original_cluster_centers[0], original_cluster_centers[1]
        except Exception as e:
            return [0,0,0], [0,0,0]
    
    def postprocess_samples(self):
        new_points = self.all_samples
        if self.normalize:
            new_points = normalize(new_points.astype(float), axis = 1, norm="l2")
        if self.scale_values:
            new_points = self.scaler.transform(self.all_samples)
        self.KMeans = KMeans(n_clusters=2, random_state=0)
        self.KMeans.fit(new_points)


    


    
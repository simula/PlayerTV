from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize, StandardScaler
import numpy as np
import cv2

from tools.outside_modules.rgb_module.distance_metrics_rgb import weighted_RGB, to_cielab, rgb
from tools.outside_modules.rgb_module.get_crop_offset import get_crop_center_offset, get_crop_trim_offset, parse_offset

color_space = {"weighted_rgb": weighted_RGB, "CIELAB": to_cielab, "rgb": rgb}
get_crop_offset = {"center": get_crop_center_offset, "trim": get_crop_trim_offset}


class RGBModule:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.rgb_function = color_space[parameters["color_space"]]
        self.normalize = parameters["normalize"]
        self.scale_values = parameters["scale_values"]
        self.certain_iou_thresh = parameters["certain_iou_thresh"]
        self.scaler = StandardScaler()

        try:
            self.get_crop = get_crop_offset[parameters["offset_type"]]
            self.offset = parse_offset(parameters["offset"])
        except:
            # If no offset, return the same crop
            print("Not provided offset, running full crop RGB version")
            self.get_crop = lambda x, offset: x
            self.offset = None

        self.all_samples = []
        self.cluster_labels = None
        self.cluster_means = None

    def process_sample(self, point):
        if self.normalize:
            point = normalize(point.astype(float), axis = 1, norm="l2")
        if self.scale_values:
            point = self.scaler.transform(point)
        return point

    def add_sample(self, sample: np.array):
        ## sample being the rgb value before normalization and scaling
        self.all_samples.append(sample)


    def get_avg_rgb(self, crop):
        crop = self.get_crop(crop, self.offset)
        avg_rgb = self.rgb_function(crop)
        return avg_rgb

    def get_cluster_labels(self):
        if not isinstance(self.cluster_labels, np.ndarray):
            self.cluster()
        return self.cluster_labels
    
    def get_cluster_means(self):
        if not isinstance(self.cluster_labels, np.ndarray):
            self.cluster()
        return self.cluster_means

    def cluster(self):
        NotImplementedError("Cluster must be implemented in subclasses.")


class DBSCANModule(RGBModule):
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.dbscan = DBSCAN(eps=self.parameters["eps"], min_samples=self.parameters["min_samples"])
        super().__init__(parameters)
    
    def get_parameters(self):
        return self.parameters
    
    def cluster(self):
        self.all_samples = np.asarray(self.all_samples)
        samples = self.all_samples
        samples = self.process_sample(samples)

        cluster_means = []
        unique_labels = np.unique(self.cluster_labels)[self.cluster_labels != 1]
        for label in unique_labels:
            cluster_points = samples[self.cluster_labels == label]
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_means.append(cluster_mean)

        cluster_means = np.asarray(cluster_means)
        self.cluster_means = cluster_means
        return np.array(self.cluster_means)

    def predict_cluster(self, point):
        try:
            point = point
            point = point.reshape(1, -1)
            point = self.process_sample(point)
            idx = 0
            distance = None
            selected_cluster = None
            for cluster_mean in self.cluster_means:
                dist = np.linalg.norm(point - cluster_mean, axis=1)
                if distance == None or dist < distance:
                    selected_cluster = idx
                    distance = dist
                idx += 1
                
            return selected_cluster
        
        except Exception as e:
            print(e)
            return None
    

class KMeansModule(RGBModule):
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.kmeans = KMeans(n_clusters=2, random_state=0)
        super().__init__(parameters)


    def get_parameters(self):
        return self.parameters
    
    def cluster(self):
        self.all_samples = np.asarray(self.all_samples)
        samples = self.all_samples
        samples = self.process_sample(samples)
        
        self.kmeans.fit(samples)

        cluster_means = self.kmeans.cluster_centers_
        self.cluster_means = cluster_means
        return np.array(self.cluster_means)

    def predict_cluster(self, point):
        try:
            point = point.reshape(1, -1)
            self.process_sample(point)
        
            cluster = self.kmeans.predict(point)
            return cluster[0]
        
        except Exception as e:
            print(e)
            return None
        
        
        

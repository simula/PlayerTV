from collections import defaultdict
import numpy as np


def plurality_vote(post_processor, samples):
    rgb_module = post_processor.rgb_module
    idx_dict = defaultdict(int)

    for sample in samples:
        cluster = rgb_module.predict_cluster(sample)
        idx_dict[cluster] += 1
        if len(samples) == 1:
            return cluster
    return_idx = max(idx_dict, key=idx_dict.get)
    return return_idx
    
def min_total_distance(post_processor, samples):
    rgb_module = post_processor.rgb_module
    cluster_means = rgb_module.get_cluster_means()
    idx_dict = defaultdict(int)

    for sample in samples:
        rgb = sample["avg_rgb"]
        for i in range(len(cluster_means)):
            dist = rgb_module.distance_metric(rgb, cluster_means[i])
            idx_dict[i] += dist

    return_idx = min(idx_dict, key=idx_dict.get)
    return return_idx
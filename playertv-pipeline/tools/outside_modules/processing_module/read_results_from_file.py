import numpy as np


class OnlineTargetFile:
    def __init__(self, txt_line) -> None:
        info = txt_line.strip().split(",")
        self.info = info
        self.frame = int(info[0])
        self.track_id = int(info[1])
        self.last_tlwh = np.array([info[2].strip(), info[3].strip(), info[4].strip(), info[5].strip()], dtype=float)
        self.score = info[6].strip()

    def __str__(self) -> str:
        return str(self.info)
    
    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.last_tlwh.copy()
        ret[2:] += ret[:2]
        return ret

class OnlineTargetLoader:
    def __init__(self, txt) -> None:
        file = open(txt, "r")
        self.file = file.readlines()
        self.index = 0

    def __call__(self):
        online_targets = []
        if self.index == len(self.file):
            return []
        sample = OnlineTargetFile(self.file[self.index])
        online_targets.append(sample)
        while True:
            if self.index + 1 == len(self.file):
                self.index += 1
                return online_targets
            sample2 = OnlineTargetFile(self.file[self.index+1])
            if sample2.frame == sample.frame:
                online_targets.append(sample2)
                self.index += 1
            else:
                sample = sample2
                self.index += 1
                return online_targets, self.get_iou_matrix(online_targets)
            
    def get_iou_matrix(self, stracks):
        if len(stracks) > 0:
            list_of_tlbr = np.array([track.tlbr for track in stracks])
            
            x_min = np.maximum(list_of_tlbr[:, 0][:, np.newaxis],list_of_tlbr[:, 0])
            y_min = np.maximum(list_of_tlbr[:, 1][:, np.newaxis],list_of_tlbr[:, 1])
            x_max = np.minimum(list_of_tlbr[:, 2][:, np.newaxis],list_of_tlbr[:, 2])
            y_max = np.minimum(list_of_tlbr[:, 3][:, np.newaxis],list_of_tlbr[:, 3])

            intersection_area = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
            area_box1 = (list_of_tlbr[:, 2] - list_of_tlbr[:, 0]) * (list_of_tlbr[:, 3] - list_of_tlbr[:, 1])
        
            union_area = (area_box1[:, np.newaxis] + area_box1) - intersection_area

            iou = intersection_area / union_area
            iou[np.isnan(iou)] = 0  # Set IoU to 0 for cases where union_area is 0

            return iou
        else:
            return None
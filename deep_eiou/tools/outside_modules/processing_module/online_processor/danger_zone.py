import inspect


class DangerZone:
    memory = 10
    def __init__(self, start_frame, list_of_tracklets, iou):
        self.start_frame = start_frame
        self.tracklets = list_of_tracklets
        self.end_frame = start_frame
        self.resolved = False
        self.active = True
        self.max_iou = iou
        self.swap_frame = start_frame
        ##Keep states of tracklets when entered

    def __str__(self) -> str:
        players_str = '\n'.join(str(player) for player in self.tracklets)
        return f"{players_str}\nSTARTFRAME:{self.start_frame}, ENDFRAME: {self.end_frame}, CHANGEFRAME: {self.swap_frame}, ACTIVE: {self.active}\n\n"

    def add(self, tracklet):
        self.tracklets.append(tracklet)

    def update(self, iou, frame_num):
        if iou > self.max_iou:
            self.max_iou = iou
            self.swap_frame = frame_num
        self.end_frame = frame_num


    def check_samples(self, tracka, trackb, frame_num, iou):
        time_since_active = frame_num - self.end_frame
        if self.active:
            if time_since_active > self.memory:
                self.active = False
                return False
            if tracka in self.tracklets and trackb in self.tracklets:
                self.update(iou, frame_num)
            elif tracka in self.tracklets:
                self.add(trackb)
                self.update(iou, frame_num)
            elif trackb in self.tracklets:
                self.add(tracka)
                self.update(iou, frame_num)
            else:
                return False
            return True
        else:
            return False
        
    def check_consistency(self, txts, confs, cluster_id, mytracklet):
        consistent = []
        consistent_rgb = []
        consistent_ocr = []
        for tracklet in self.tracklets:
            rgb_is_consistent = mytracklet.rgb_is_consistent(cluster_id, other_track=tracklet)
            if rgb_is_consistent:
                consistent_rgb.append(tracklet)
            
            if txts != None:
                for i in range(len(txts)):
                    txt = txts[i]
                    conf = confs[i]
                    if conf > 0.5:
                        ocr_is_consistent = mytracklet.ocr_is_consistent(txt, other_track=tracklet)
                        if ocr_is_consistent:
                            consistent_ocr.append(tracklet)
        if len(consistent) != 0:
            print("Found consistent tracklets")
            for i in consistent_ocr:
                print(i)               

    def visited(self, frame_num, iou):
        self.end_frame = frame_num
        self.frames_since_active = 0
        if iou > self.max_iou:
            self.swap_frame = frame_num
            self.max_iou = iou

    def not_visited(self):
        self.active_in_frame = False
        self.frames_since_active += 1

class IdSwitch(DangerZone):
    def __init__(self, start_frame, list_of_tracklets, iou):
        super().__init__(start_frame, list_of_tracklets, iou)



class KitNumberClash(DangerZone):
    def __init__(self, start_frame, list_of_tracklets, team_idx, kit_number):
        super().__init__(start_frame, list_of_tracklets)
        self.kit_number = kit_number
        self.team_idx = team_idx




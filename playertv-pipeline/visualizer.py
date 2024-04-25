import os.path as osp
import cv2
import numpy as np

from post_processing.smart_crop_detector import Smart_Crop_Detector, Sample_Crop


class Visualizer:
    def __init__(self):
        self.team_colors = [(255, 0, 0), (0, 0, 255), (0,0,0)]
        self.color_txt = (250, 250, 250)
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_size = 0.3

    def _draw_rectangle(self, sample, frame, team_idx, player=None):
        ## Draw rectangle
        text = f"{sample.ocr} ID: {sample.id}"
        x = int(sample.xywh[0])
        y = int(sample.xywh[1])
        w = int(sample.xywh[2])
        h = int(sample.xywh[3])

        top_left = (x, y)
        bot_right = (x+w, y+h)

        (w_text, h_text), _ = cv2.getTextSize(text, self.font, self.font_size, 1)
        cv2.rectangle(frame, (top_left[0], top_left[1]-h_text-1), (top_left[0]+w_text, top_left[1]-1), self.team_colors[team_idx], -1)
        frame = cv2.putText(frame, text, (top_left[0], top_left[1]-1), self.font, self.font_size, (250,250,250), 1)
        frame = cv2.rectangle(frame, top_left, bot_right, color=self.team_colors[team_idx], thickness=1)

        
        return frame
    
    def _return_crop_frame(self, size: tuple, sample, frame):
        ## Draw rectangle
        frame_size = frame.shape[:2]
        size_y = size[0]
        size_x = size[0]

        text = f"{sample.ocr}"
        x = int(sample.xywh[0])
        y = int(sample.xywh[1])
        w = int(sample.xywh[2])
        h = int(sample.xywh[3])

        center = (y+h/2, x+w/2)
        crop_tl_y = int(center[0] - size_y/2)
        crop_tl_x = int(center[1] - size_x/2)
        crop_br_y = int(center[0] + size_y/2)
        crop_br_x = int(center[1] + size_x/2)

        i = 0
        while crop_br_y - crop_tl_y < size_y:
            if i % 2 == 0:
                crop_br_y += 1
            else:
                crop_tl_y -= 1
            i += 1

        i = 0
        while crop_br_x - crop_tl_x < size_x:
            if i % 2 == 0:
                crop_br_x += 1
            else:
                crop_tl_x -= 1
            i += 1

        y_start = max(crop_tl_y, 0)
        x_start = max(crop_tl_x, 0)
        y_end = min(crop_br_y, frame_size[0])
        x_end = min(crop_br_x, frame_size[1])
        pad_top = max(0 - crop_tl_y, 0)
        pad_bot = max(crop_br_y - frame_size[0], 0)
        pad_left = max(0 - crop_tl_x, 0)
        pad_right = max(crop_br_x - frame_size[1], 0)

        top_left = (x, y)
        bot_right = (x+w, y+h)

        (w_text, h_text), _ = cv2.getTextSize(text, self.font, self.font_size, 1)
        cv2.rectangle(frame, (top_left[0], top_left[1]-h_text-1), (top_left[0]+w_text, top_left[1]-1), self.team_colors[0], -1)
        frame = cv2.putText(frame, text, (top_left[0], top_left[1]-1), self.font, self.font_size, (250,250,250), 1)
        frame = cv2.rectangle(frame, top_left, bot_right, color=self.team_colors[0], thickness=1)
        cropped_image = frame[y_start:y_end, x_start:x_end]
        padded_image = np.pad(cropped_image, ((pad_top, pad_bot), (pad_left, pad_right), (0, 0)), mode='constant')
        return padded_image
    

    def visualize_player(self, video_path, label_file, player, size):
        file_name = osp.basename(video_path)
        save_folder = osp.dirname(label_file)
        print("Save destination =", save_folder)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file. Do you have the correct path?\nPath provided: {video_path}")
        
        print("file_name:", file_name)
        
        frame_rate = int(cap.get(5))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        delay = int(1000/frame_rate)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{save_folder}/{file_name}", fourcc, frame_rate, size)
        ret, frame = cap.read()
        if not ret:
            raise Exception("Not able to return the next frame, check your VideoCapture")
        
        i = 0
        c = 0
        with open(label_file, "r") as file:
            for line in file:
                sample = Sample_Crop(line)

                while sample.frame != i:
                    #cv2.imshow("YOLOX Tracking", frame)
                    #if cv2.waitKey(delay) & 0xFF == ord('q'):
                    #    break

                    ret, frame = cap.read()
                    if not ret:
                        raise Exception("Not able to return the next frame, check your VideoCapture")
                    i +=1
                if player in sample.ocr.strip().split():
                    frame = self._return_crop_frame(size, sample, frame)
                    out.write(frame)
    
        cap.release()
        out.release()
        cv2.destroyAllWindows()


    def visualize_kit_team(self, video_path, label_file, player=None, custom_save_path = None, alt_out_name = None):
        file_name = osp.basename(video_path)
        if custom_save_path:
            save_folder = custom_save_path
        else:
            save_folder = osp.dirname(label_file)
        print("Save destination =", save_folder)
        team_map = {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file. Do you have the correct path?\nPath provided: {video_path}")
        
        print("file_name:", file_name)
        
        frame_rate = int(cap.get(5))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        delay = int(1000/frame_rate)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #Alternative output name
        if alt_out_name:
            out = cv2.VideoWriter(f"{save_folder}/{alt_out_name}_{file_name}", fourcc, frame_rate, (frame_width, frame_height))
        else:
            out = cv2.VideoWriter(f"{save_folder}/{file_name}", fourcc, frame_rate, (frame_width, frame_height))

        ret, frame = cap.read()
        if not ret:
            raise Exception("Not able to return the next frame, check your VideoCapture")
        
        i = 0
        c = 0
        with open(label_file, "r") as file:
            for line in file:
                sample = Sample_Crop(line)
                try:
                    team_idx = team_map[sample.team]
                except:
                    team_map[sample.team] = c + 1
                    c += 1
                    team_idx = c
                sample.team

                while sample.frame != i:
                    out.write(frame)
                    #cv2.imshow("YOLOX Tracking", frame)
                    #if cv2.waitKey(delay) & 0xFF == ord('q'):
                    #    break

                    ret, frame = cap.read()
                    if not ret:
                        raise Exception("Not able to return the next frame, check your VideoCapture")
                    i +=1
                if player:
                    if player in sample.ocr.strip().split():
                        frame = self._draw_rectangle(sample, frame, team_idx, player)
                else:
                    frame = self._draw_rectangle(sample, frame, team_idx, player)
    
        cap.release()
        out.release()
        cv2.destroyAllWindows()

                


if __name__ == "__main__":
    visualizer = Visualizer()

    video_folder = "../../videos"
    video_path = osp.join(video_folder, "3783_afwx4di2qyfjy.mp4")

    label_folder = "results/3783_afwx4di2qyfjy/2024_02_26_20_52_04"
    
    label_file = osp.join(label_folder, "final_results.txt")

    ocr_results = "test/smart_crop/ocr_results.json"
    #ocr_results = osp.join(label_file, "ocr_results.json")
    #visualizer.visualize_kit_team(video_path, label_file, "Åsen")
    #visualizer.visualize_player(video_path, label_file, "Åsen", (180, 320))
    visualizer.visualize_kit_team(video_path, label_file)
import cv2
import os
import glob
from PIL import Image

class ImageFolderCapture:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.images = sorted(glob.glob(os.path.join(folder_path, '*')))
        self.index = 0
        self.length = len(self.images)
        with Image.open(self.images[0]) as img:
            self.width, self.height = img.size
            self.size = img.size
        
    def read(self):
        if self.index < self.length:
            image = cv2.imread(self.images[self.index])
            self.index += 1
            return True, image
        else:
            return False, None
    
    def isOpened(self):
        return self.index < self.length
    
    def release(self):
        self.index = self.length
    
    def set(self, propId, value):
        if propId == cv2.CAP_PROP_POS_FRAMES:
            self.index = int(value)
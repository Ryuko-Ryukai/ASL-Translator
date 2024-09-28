import cv2
from time import time

class fps:
    def __init__(self)->None:
        self.fps = 0
        self.prev_frame_time = 0
        self.new_frame_time = 0
    
    def fpsCal(self)->float:
        self.new_frame_time = time()
        self.fps = 1/(self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time

    def FPS_FRONT_CAM_SHOW(self, img)->None:
        cv2.putText(img, f'FPS: {int(self.fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    def FPS_SELF_CAM_SHOW(self, img)->None:
        img = cv2.flip(img, 1)
        cv2.putText(img, f'FPS: {int(self.fps)}', (10, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 0 ,0), 2, cv2.LINE_AA)

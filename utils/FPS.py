import cv2
from time import time

class fps:
    def __init__(self)->None:
        self.__fps = 0
        self.__prev_frame_time = 0
        self.__new_frame_time = 0
    
    def fpsCal(self)->float:
        self.__new_frame_time = time()
        self.__fps = 1/(self.__new_frame_time - self.__prev_frame_time)
        self.__prev_frame_time = self.__new_frame_time

    def FPS_FRONT_CAM_SHOW(self, img)->None:
        """
        Show the img capture from the front camera

        :param img: The image which is captured from webcam
        """

        cv2.putText(img, f'FPS: {int(self.__fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    def FPS_SELF_CAM_SHOW(self, img)->None:
        """
        Show the img capture from the self camera

        :param img: The image which is captured from webcam
        """

        img = cv2.flip(img, 1)
        cv2.putText(img, f'FPS: {int(self.__fps)}', (10, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 0 ,0), 2, cv2.LINE_AA)

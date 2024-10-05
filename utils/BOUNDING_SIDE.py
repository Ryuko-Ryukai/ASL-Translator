import cv2
import numpy as np

class bbox:
    def __init__(self)->None:
        self.__y_min = 0
        self.__y_max = 0
        self.__x_min = 0
        self.__x_max = 0
    
    def bbox_draw(self, img, hand_landmarks, offset:int=20)->None:
        """
        Show the img capture from the front camera

        :param img: The image which is captured from webcam
        :param hand_landmarks: the coordinates of hand tracking in mediapipe
        """

        self.__x_min = min([landmark.x for landmark in hand_landmarks.landmark])
        self.__x_max = max([landmark.x for landmark in hand_landmarks.landmark])
        self.__y_min = min([landmark.y for landmark in hand_landmarks.landmark])
        self.__y_max = max([landmark.y for landmark in hand_landmarks.landmark])
        self.__x_min = int(self.__x_min * img.shape[1]) - offset
        self.__x_max = int(self.__x_max * img.shape[1]) + offset
        self.__y_min = int(self.__y_min * img.shape[0]) - offset
        self.__y_max = int(self.__y_max * img.shape[0]) + offset

    def bbox_show(self, img, label=None)->None:
        """
        Show the img capture from the front camera

        :param img: The image which is captured from webcam
        :param label: The result when detect right hand and left hand (["Left", "Right"])
        """
        cv2.rectangle(img, (self.__x_min, self.__y_min), (self.__x_max, self.__y_max), (0, 255, 0), 2)
        cv2.putText(img, label, (self.__x_min, self.__y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    @staticmethod
    def bbox_coord(img, hand_landmarks)->tuple[int, int, int, int]:
        """
        Show the img capture from the front camera

        :param img: The image which is captured from webcam
        param hand_landmarks: the coordinates of hand tracking in mediapipe
        """
        img_height, img_width, _ = img.shape
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]

        x_coords = np.array(x_coords) * img_width
        y_coords = np.array(y_coords) * img_height

        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))

        width = x_max - x_min
        height = y_max - y_min

        return x_min, y_min, width, height

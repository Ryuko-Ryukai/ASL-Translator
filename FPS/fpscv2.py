from collections import deque
import cv2

class fps(object):
    def __init__(self, buffer_len=1)->None:
        self._tick = cv2.getTickCount()
        self._freq = 1000.0/cv2.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def getter(self)->float:
        current_tick = cv2.getTickCount()
        diff_time = (current_tick-self._tick)*self._freq
        self._tick = current_tick

        self._difftimes.append(diff_time)

        fps = 1000.0/(sum(self._difftimes)/len(self._difftimes))
        fps_fix = round(fps, 1)

        return fps_fix
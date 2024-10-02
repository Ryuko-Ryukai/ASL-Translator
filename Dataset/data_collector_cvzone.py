import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
import math as m
import time

folder = 'path/to/dataset'
count = 0

# Initialize the webcam
# 0 selects the default camera (usually the built-in webcam).
# 1 selects the first external camera.
# 2 selects the second external camera, and so on.
cap = cv2.VideoCapture(0)

# Initialize the hand detector with a maximum of 2 hands
detector = HandDetector(maxHands=2)

# Define the offset and image size for the cropped hand image
offset = 20
imgSize = 300

while True:
    # Capture frame-by-frame from the webcam
    # img: This is the original image captured from the webcam.
    # This is a boolean value. It will be True if the frame was successfully read, and False otherwise.
    result, img = cap.read()
    
    # Detect hands in the frame
    hands, img = detector.findHands(img)
    # hands: This is a list of dictionaries, where each dictionary contains information about a detected hand. The information typically includes:
        # bbox: Bounding box coordinates of the hand.
        # lmList: List of landmarks (key points) on the hand.
        # center: Center point of the hand.
        # type: Type of hand (left or right).
    # img: The input image with annotations (such as landmarks and bounding boxes) drawn on it.
    
    if hands:
        # Get the bounding box of the first detected hand
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white image of size 300x300
        # np.uint8 stands for unsigned 8-bit integer. This means each pixel value in the 
        # image will be an integer between 0 and 255.

        #Using np.uint8 is standard for images because it efficiently represents the pixel 
        # values without using too much memory.

        # np.ones((imgSize, imgSize, 3), np.uint8) creates an array of ones with the shape (300, 300, 3), 
        # where 3 represents the three color channels (Red, Green, Blue).

        
        # Multiplying by 255 converts all the ones to 255, which is the maximum value for an 8-bit pixel. 
        # This results in a white image because in the RGB color model, (255, 255, 255) represents white.

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure the coordinates are within the image bounds
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

        # Crop the hand region from the frame
        imgCrop = img[y1:y2, x1:x2] # define the rectangular region to be cropped.
        # y1: The starting y-coordinate (top edge of the crop).
        # y2: The ending y-coordinate (bottom edge of the crop).
        # x1: The starting x-coordinate (left edge of the crop).
        # x2: The ending x-coordinate (right edge of the crop).

        imgCropShape = imgCrop.shape

        if imgCropShape[0] > 0 and imgCropShape[1] > 0:
            # Calculate the aspect ratio of the hand
            aspectRatios = h / w
            
            if aspectRatios > 1:
                # If height is greater than width
                k = imgSize / h
                wCal = m.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = m.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                # If width is greater than height
                k = imgSize / w
                hCal = m.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = m.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Display the white image with the resized hand
            cv2.imshow("ImageWhite", imgWhite)

            # imgWhite (300x300)
            # +---------------------+
            # |                     |
            # |      hGap (50)      |
            # | +---------------+   |
            # | |   imgResize   |   |
            # | |   (300x200)   |   |
            # | +---------------+   |
            # |      hGap (50)      |
            # |                     |
            # +---------------------+

    # Display the original frame with hand detection
    cv2.imshow("Image", img)
    
    # Wait for 1 millisecond before moving to the next frame
    key = cv2.waitKey(1)
    if key == ord("s"):
        count += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(count)
    elif key == ord('q'):
        exit()
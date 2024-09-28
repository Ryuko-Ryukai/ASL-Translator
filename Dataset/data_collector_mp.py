import numpy as np
import cv2
import mediapipe as mp
import math as m
import time
from utils import BOUNDING_SIDE

bbox = BOUNDING_SIDE.bbox()
folder = 'path/to/dataset'
count = 0

# Initialize the webcam
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define the offset and image size for the cropped hand image
offset = 40
imgSize = 300

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Draw bounding box
                bbox.bbox_draw(img=image, hand_landmarks=hand_landmarks)
                bbox.bbox_show(img=image, label=label)

                # Get bounding box coordinates
                x, y, w, h = bbox.bbox_coord(img=image, hand_landmarks=hand_landmarks)

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                # Ensure the coordinates are within the image bounds
                y1, y2 = max(0, y - offset), min(image.shape[0], y + h + offset)
                x1, x2 = max(0, x - offset), min(image.shape[1], x + w + offset)

                # Crop the hand region from the frame
                imgCrop = image[y1:y2, x1:x2]

                imgCropShape = imgCrop.shape

                if imgCropShape[0] > 0 and imgCropShape[1] > 0:
                    # Calculate the aspect ratio of the hand
                    aspectRatios = h / w
                    
                    if aspectRatios > 1:
                        # If height is greater than width
                        k = imgSize / h
                        wCal = m.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = m.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        # If width is greater than height
                        k = imgSize / w
                        hCal = m.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = m.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    # Display the white image with the resized hand
                    cv2.imshow("ImageWhite", imgWhite)

        # Display the original frame with hand detection
        cv2.imshow("Image", image)
        
        # Wait for 1 millisecond before moving to the next frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            count += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(count)
        elif key == 27:
            exit()

cap.release()
cv2.destroyAllWindows()
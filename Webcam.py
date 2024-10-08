import cv2
import mediapipe as mp
from utils import FPS, BOUNDING_SIDE

class webcam:
    def __init__(self) -> None:
        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_drawing_styles = mp.solutions.drawing_styles
        self.__mp_hands = mp.solutions.hands

    def live_vid(self)->None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FPS, 60)

        fps = FPS.fps()
        bbox = BOUNDING_SIDE.bbox()

        with self.__mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                ret, img = cap.read()
                if not ret:
                    print("Ignore empty webcam")
                    continue
                
                # Flip because webcam will showing the inverse-frame video
                img = cv2.flip(img, 1)
                img.flags.writeable=False
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img)

                img.flags.writeable=True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        #labels = handedness.classification[0].label
                        #print(f'Landmarks: {hand_landmarks}')

                        self.__mp_drawing.draw_landmarks(
                            img,
                            hand_landmarks,
                            self.__mp_hands.HAND_CONNECTIONS,
                            self.__mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.__mp_drawing_styles.get_default_hand_connections_style())
                        
                        bbox.bbox_draw(img=img, hand_landmarks=hand_landmarks)
                        #bbox.bbox_show(img=img, label=labels)
                        bbox.bbox_show(img=img)

                fps.fpsCal()
                fps.FPS_FRONT_CAM_SHOW(img=img)
                cv2.imshow("Wecam", img)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    exit()

    def static_img(self, image_files) -> None:
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
            for idx, file in enumerate(image_files):
                image = cv2.flip(cv2.imread(file), 1)
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                print('Handedness:', results.multi_handedness)
                if not results.multi_hand_landmarks:
                    continue
                image_height, image_width, _ = image.shape
                annotated_image = image.copy()
                for hand_landmarks in results.multi_hand_landmarks:
                    print('hand_landmarks:', hand_landmarks)
                    print(
                        f'Index finger tip coordinates: (',
                        f'{hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                        f'{hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                    )
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                cv2.imwrite(
                    f'/tmp/annotated_image_{idx}.png', cv2.flip(annotated_image, 1))
                if not results.multi_hand_world_landmarks:
                    continue
                for hand_world_landmarks in results.multi_hand_world_landmarks:
                    self.mp_drawing.plot_landmarks(
                        hand_world_landmarks, self.mp_hands.HAND_CONNECTIONS, azimuth=5)
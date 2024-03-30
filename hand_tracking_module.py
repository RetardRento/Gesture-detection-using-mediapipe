import cv2 as cv
import mediapipe as mp

class HandTracking:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.hands = mp.solutions.hands.Hands(static_image_mode=self.static_image_mode, 
                                              max_num_hands=self.max_num_hands,
                                              min_detection_confidence=self.min_detection_confidence,
                                              min_tracking_confidence=self.min_tracking_confidence)
        self.draw = mp.solutions.drawing_utils

    def find_hands(self, image, draw=True):
        self.result = self.hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        if self.result.multi_hand_landmarks:
            for landmarks in self.result.multi_hand_landmarks:
                self.draw.draw_landmarks(image, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                if draw:
                    cv.imshow("video", image)
        return image

    def landmark_tracker(self, image, handno=0, draw=True):
        self.find_hands(image, draw=draw)
        h, w, c = image.shape
        lmlist = []
        if self.result.multi_hand_landmarks:
            for id, lm in enumerate(self.result.multi_hand_landmarks[handno].landmark):
                xc, yc = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, xc, yc])
        if len(lmlist) != 0:
            return lmlist

    def close(self):
        cv.destroyAllWindows()


def create_hand_tracker():
    return HandTracking()

if __name__ == "__main__":
    video = cv.VideoCapture(0)
    HandTracker = HandTracking()
    while True:
        is_true, frame = video.read()
        image = HandTracker.landmark_tracker(image=frame, handno=0)
        print(image)
        if cv.waitKey(1) != -1:
            break
    HandTracker.close()

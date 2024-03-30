from flask import Flask, render_template, Response
import cv2
import mediapipe as mp 

app = Flask(__name__)

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
        self.result = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self.result.multi_hand_landmarks:
            for landmarks in self.result.multi_hand_landmarks:
                self.draw.draw_landmarks(image, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                if draw:
                    cv2.imshow("video", image)
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
        cv2.destroyAllWindows()

def generate_frames():
    video = cv2.VideoCapture(0)
    hand_tracker = HandTracking()

    if not video.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        is_true, frame = video.read()
        if not is_true:
            print("Error: Could not read frame.")
            break

        image = hand_tracker.landmark_tracker(image=frame, handno=0)

        # Convert the image to JPEG format
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    hand_tracker.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

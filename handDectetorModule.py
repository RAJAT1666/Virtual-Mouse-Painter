import cv2 as cv
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.maxHands = max_hands
        self.trackingConfidence = tracking_confidence
        self.detectionConfidence = detection_confidence

        # Initialization
        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(static_image_mode=self.mode,
                                       max_num_hands=self.maxHands,
                                       min_tracking_confidence=self.trackingConfidence,
                                       min_detection_confidence=self.detectionConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    # Return the landmarks of the hand.
    def getHand(self, img, draw=False):
        imgRgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.hands.process(imgRgb)
        lst = []
        # Results has landmarks for every hand.
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                for idx, land in enumerate(hand.landmark):
                    y, x, z = img.shape
                    cx = int(x * land.x)
                    cy = int(y * land.y)
                    lst.append([idx, cx, cy])
                self.mpDraw.draw_landmarks(img, hand, self.mpHand.HAND_CONNECTIONS)
        if draw:
            self.drawPoint(img, lst)
        return lst

    def drawPoint(self, img, lst, clr=(0, 0, 255), r=7):
        if lst:
            for idx, x, y in lst:
                cv.circle(img, (x, y), r, clr, -1)


def starter(webcam_number=0):
    cap = cv.VideoCapture(webcam_number)   # Capture Video
    detector = HandDetector()
    lastTime = 0
    while True:
        success, img = cap.read()
        img = cv.flip(img, 1)
        lst = detector.getHand(img)

        # FPS
        cTime = time.time()
        fps = 1 // (cTime - lastTime)
        lastTime = cTime
        cv.putText(img, str(fps), (10, 69), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

        cv.imshow('img', img)
        cv.waitKey(1)


if __name__ == '__main__':
    starter()

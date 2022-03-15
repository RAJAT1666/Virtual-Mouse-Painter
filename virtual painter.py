import cv2
import cv2 as cv
import mediapipe as mp
import time, os
import handDectetorModule as hdm
import numpy as np

# Location of image to use in header
path = r'header'
all_image_path = [path + '/' + i for i in os.listdir(path)]  # Storing Path
all_image = [cv.imread(i) for i in all_image_path]  # Storing image

# Initialization
detector = hdm.HandDetector()
cap = cv.VideoCapture(0)
lastHeader = 0
colorList = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 0, 0)]  # B, R, G, W
lastX, lastY = -1, -1
canvas = np.zeros((480, 640, 3), np.uint8)
brush = 10
eraser = 20
sizeList = [brush, brush, brush, eraser]


# Function to put the desired image as header

def put(img, idx):
    img2 = all_image[idx]
    imy, imx, imz = img.shape
    img2 = cv.resize(img2, (imx, 100), interpolation=cv.INTER_LINEAR)
    img[:100, :imx] = img2


# Function to count the number of finger that are up.
def count_fingers(lst):
    totalFinger = 0
    fingerUp = []
    if lst:
        totalFinger = 5
        # Upper -> 4, 8, 12 ,16, 20
        # nxt -> 3, 7, 11, 15, 19
        # nxt -> 2, 6, 10, 14, 18
        up = [4, 8, 12, 16, 20]
        dn = [3, 7, 11, 15, 19]
        nxt = [2, 6, 10, 14, 18]
        # For thumb
        if min(lst[3][1], lst[17][1]) <= lst[4][1] <= max(lst[3][1], lst[17][1]):
            totalFinger -= 1
            fingerUp.append(0)
        else:
            fingerUp.append(1)
        for i, j, k in zip(up[1:], dn[1:], nxt[1:]):
            if lst[i][2] >= lst[j][2] or lst[i][2] >= lst[k][2]:
                totalFinger -= 1
                fingerUp.append(0)
            else:
                fingerUp.append(1)
    return totalFinger, fingerUp


while True:
    success, img = cap.read()
    img = cv.flip(img, 1)

    lst = detector.getHand(img)
    totalFinger, fingerUp = count_fingers(lst)

    # Drawing mode
    if totalFinger == 1 and fingerUp[1]:
        detector.drawPoint(img, [lst[8]], colorList[lastHeader], sizeList[lastHeader])
        idx, x, y = lst[8]

        # Checking if it is the first frame or not
        if lastX == lastY == -1:
            lastX = x
            lastY = y
        cv.line(img, (lastX, lastY), (x, y), colorList[lastHeader], sizeList[lastHeader])
        cv.line(canvas, (lastX, lastY), (x, y), colorList[lastHeader], sizeList[lastHeader])

        # Updating the last location
        lastX = x
        lastY = y

    # Selection mode
    else:
        lastX = -1
        lastY = -1
        if totalFinger > 1 and fingerUp[1]:
            idx, x, y = lst[8]

            # Deciding which function is selected
            if y < 100:
                if 0 < x < 130:
                    lastHeader = 0
                if 185 < x < 300:
                    lastHeader = 1
                if 340 < x < 450:
                    lastHeader = 2
                if 500 < x:
                    lastHeader = 3


    # SuperImposing canvas onto image.
    imgGray = cv2.cvtColor(canvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)     # Black color where we have drawn something.
    img = cv.bitwise_and(img, imgInv)   # Draw a black color on the image where we have drawn something.
    img = cv.bitwise_or(img, canvas)    # Imposing color on black.

    put(img, lastHeader)
    cv.imshow('img', img)
    cv.imshow('canvas', canvas)
    cv.waitKey(1)

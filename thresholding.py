import numpy as np
import cv2

# Threshold for the Sobel X gradient (computed on the grayscale image)
gradient_threshold = 80

# HSV color range for the yellow lane lines
lower_yellow = np.array([15, 100, 100])
upper_yellow = np.array([35, 255, 255])

# BGR color range for the white lane lines
lower_white = np.array([200, 200, 200])
upper_white = np.array([255, 255, 255])


def threshold_image(img):
    # BGR to Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel in X direction
    sobelX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    absX = cv2.convertScaleAbs(sobelX)
    ret, gradientMask = cv2.threshold(absX, gradient_threshold, 255, cv2.THRESH_BINARY)

    # BGR to HSV: yellow color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellowMask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # White color
    whiteMask = cv2.inRange(img, lower_white, upper_white)
    colorMask = cv2.bitwise_or(yellowMask, whiteMask)

    # Combine the masks
    binary = cv2.bitwise_or(gradientMask, colorMask)

    return binary

import numpy as np
import cv2

# Source points (trapezoid on the road) and destination points (rectangle in the bird's-eye view)
# Order: top left, top right, bottom right, bottom left
PERSPECTIVE_POINTS = {
    (1280, 720): (
        np.float32([[582, 460], [703, 460], [1106, 720], [213, 720]]),
        np.float32([[320, 0], [960, 0], [960, 720], [320, 720]]),
    ),
    (960, 540): (
        np.float32([[426, 345], [542, 345], [848, 540], [148, 540]]),
        np.float32([[240, 0], [720, 0], [720, 540], [240, 540]]),
    ),
}


def get_perspective_points(img):
    # Pick the points for the size of the image
    return PERSPECTIVE_POINTS[(img.shape[1], img.shape[0])]


def warper(img, src, dst):
    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

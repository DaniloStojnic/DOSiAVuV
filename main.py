import numpy as np
import cv2
import os

from calibration import calibrate_camera, undistort
from thresholding import threshold_image
from perspective import get_perspective_points, warper


def main():
    os.makedirs('output_images', exist_ok=True)

    # Calibrate the camera and print the results
    mtx, dist = calibrate_camera(show=True)
    print('Camera matrix:\n', mtx)
    print('Distortion coefficients:\n', dist)

    # Save the examples of distortion correction for the writeup (chess board and road image)
    examples = [
        ('input_images/camera_cal/calibration1.jpg', 'output_images/undistort_calibration1.jpg'),
        ('input_images/test_images/test1.jpg', 'output_images/undistort_test1.jpg'),
    ]

    for src, dst in examples:
        img = cv2.imread(src)
        undistortedImg = undistort(img, mtx, dist)
        result = np.hstack((img, undistortedImg))
        cv2.imwrite(dst, result)

        # Display the final result
        cv2.imshow('original | undistorted', cv2.resize(result, None, fx=0.5, fy=0.5))

        # Wait until the window is closed before moving to the next image
        while cv2.getWindowProperty('original | undistorted', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    # Create the thresholded binary image (color + gradient) and save the example for the writeup
    img = undistort(cv2.imread('input_images/test_images/test1.jpg'), mtx, dist)
    binary = threshold_image(img)
    result = np.hstack((img, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)))
    cv2.imwrite('output_images/binary_test1.jpg', result)

    # Display the final result
    cv2.imshow('undistorted | binary', cv2.resize(result, None, fx=0.5, fy=0.5))

    # Wait until the window is closed before moving to the next step
    while cv2.getWindowProperty('undistorted | binary', cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(100)

    cv2.destroyAllWindows()

    # Perspective transform (bird's-eye view) on the straight road
    img = undistort(cv2.imread('input_images/test_images/straight_lines1.jpg'), mtx, dist)
    src, dst = get_perspective_points(img)
    warped = warper(img, src, dst)

    # Draw the source trapezoid and the destination rectangle and save the example for the writeup
    cv2.polylines(img, [src.astype(np.int32)], True, (0, 0, 255), 3)
    cv2.polylines(warped, [dst.astype(np.int32)], True, (0, 0, 255), 3)
    result = np.hstack((img, warped))
    cv2.imwrite('output_images/warped_straight_lines1.jpg', result)

    # Display the final result
    cv2.imshow('undistorted | warped', cv2.resize(result, None, fx=0.5, fy=0.5))

    # Wait until the window is closed before moving to the next step
    while cv2.getWindowProperty('undistorted | warped', cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(100)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

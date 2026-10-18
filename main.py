import numpy as np
import cv2
import os

from calibration import calibrate_camera, undistort


def main():
    os.makedirs('output_images', exist_ok=True)

    # Calibrate the camera and print the results
    mtx, dist = calibrate_camera(show=True)
    print('Camera matrix:\n', mtx)
    print('Distortion coefficients:\n', dist)

    # Save the examples of distortion correction for the writeup (chess board and road image)
    examples = [
        ('data/camera_cal/calibration1.jpg', 'output_images/undistort_calibration1.jpg'),
        ('data/test_images/test1.jpg', 'output_images/undistort_test1.jpg'),
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


if __name__ == '__main__':
    main()

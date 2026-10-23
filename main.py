import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from calibration import calibrate_camera, undistort
from thresholding import threshold_image
from perspective import get_perspective_points, warper
from lane_detection import get_histogram, fit_polynomial
from curvature import measure_curvature, measure_vehicle_offset
from drawing import draw_lane, draw_measurements


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

    # Create the thresholded binary image and save the example for the writeup
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

    # Detect the lane line pixels and fit them with a polynomial
    img = undistort(cv2.imread('input_images/test_images/test3.jpg'), mtx, dist)
    src, dst = get_perspective_points(img)
    binaryWarped = warper(threshold_image(img), src, dst)

    # Histogram of the bottom half of the warped binary image
    histogram = get_histogram(binaryWarped)
    plt.figure()
    plt.title('Histogram of the bottom half of the warped binary image')
    plt.xlabel('Column (x)')
    plt.ylabel('# of white pixels')
    plt.plot(histogram)
    plt.xlim([0, len(histogram)])
    plt.show()

    # Sliding windows and the second order polynomial fit, save the example for the writeup
    leftFit, rightFit, result = fit_polynomial(binaryWarped)
    print('Left line fit (A, B, C): ', leftFit)
    print('Right line fit (A, B, C): ', rightFit)
    cv2.imwrite('output_images/color_fit_lines_test3.jpg', result)

    # Radius of curvature of the lane and the position of the vehicle with respect to the lane center
    leftRadius, rightRadius = measure_curvature(leftFit, rightFit, binaryWarped.shape)
    offset = measure_vehicle_offset(leftFit, rightFit, binaryWarped.shape)
    print('Radius of curvature: left {:.0f} m, right {:.0f} m, mean {:.0f} m'.format(leftRadius, rightRadius, (leftRadius + rightRadius) / 2))
    print('Vehicle offset from the lane center: {:+.2f} m'.format(offset))

    # Display the final result
    cv2.imshow('sliding windows and polynomial fit', cv2.resize(result, None, fx=0.5, fy=0.5))

    # Wait until the window is closed before moving to the next step
    while cv2.getWindowProperty('sliding windows and polynomial fit', cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(100)

    cv2.destroyAllWindows()

    # Warp the detected lane back onto the original image and write the measurements, save the example for the writeup
    result = draw_lane(img, binaryWarped, leftFit, rightFit, src, dst)
    result = draw_measurements(result, (leftRadius + rightRadius) / 2, offset)
    cv2.imwrite('output_images/example_output_test3.jpg', result)

    # Display the final result
    cv2.imshow('lane on the original image', cv2.resize(result, None, fx=0.5, fy=0.5))

    # Wait until the window is closed before moving to the next step
    while cv2.getWindowProperty('lane on the original image', cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(100)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

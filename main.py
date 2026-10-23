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
from video import process_video


def main():
    os.makedirs('output_images', exist_ok=True)

    # Calibrate the camera and print the results
    mtx, dist = calibrate_camera(show=True)
    print('Camera matrix:\n', mtx)
    print('Distortion coefficients:\n', dist)

    # Save the example of distortion correction on the chess board
    img = cv2.imread('input_images/camera_cal/calibration1.jpg')
    result = np.hstack((img, undistort(img, mtx, dist)))
    cv2.imwrite('output_images/undistort_calibration1.jpg', result)

    # Display the final result
    cv2.imshow('original | undistorted', cv2.resize(result, None, fx=0.5, fy=0.5))

    # Wait until the window is closed before moving to the next step
    while cv2.getWindowProperty('original | undistorted', cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(100)

    cv2.destroyAllWindows()

    # Remove the distortion from the road image and save the example
    img = cv2.imread('input_images/test_images/test1.jpg')
    undistortedImg = undistort(img, mtx, dist)
    result = np.hstack((img, undistortedImg))
    cv2.imwrite('output_images/undistort_test1.jpg', result)

    # Display the final result
    cv2.imshow('original | undistorted', cv2.resize(result, None, fx=0.5, fy=0.5))

    # Wait until the window is closed before moving to the next step
    while cv2.getWindowProperty('original | undistorted', cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(100)

    cv2.destroyAllWindows()

    # Create the thresholded binary image from the undistorted image of the previous step and save the example
    binary = threshold_image(undistortedImg)
    result = np.hstack((undistortedImg, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)))
    cv2.imwrite('output_images/binary_test1.jpg', result)

    # Display the final result
    cv2.imshow('undistorted | binary', cv2.resize(result, None, fx=0.5, fy=0.5))

    # Wait until the window is closed before moving to the next step
    while cv2.getWindowProperty('undistorted | binary', cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(100)

    cv2.destroyAllWindows()

    # Perspective transform (bird's-eye view) of the binary image on the straight road
    img = undistort(cv2.imread('input_images/test_images/straight_lines1.jpg'), mtx, dist)
    binary = threshold_image(img)
    src, dst = get_perspective_points(binary)
    warped = warper(binary, src, dst)

    # Draw the source trapezoid and the destination rectangle and save the example
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    cv2.polylines(binary, [src.astype(np.int32)], True, (0, 0, 255), 3)
    cv2.polylines(warped, [dst.astype(np.int32)], True, (0, 0, 255), 3)
    result = np.hstack((binary, warped))
    cv2.imwrite('output_images/warped_straight_lines1.jpg', result)

    # Display the final result
    cv2.imshow('binary | warped', cv2.resize(result, None, fx=0.5, fy=0.5))

    # Wait until the window is closed before moving to the next step
    while cv2.getWindowProperty('binary | warped', cv2.WND_PROP_VISIBLE) >= 1:
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

    # Sliding windows and the second order polynomial fit, save the example
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

    # Warp the detected lane back onto the original image and write the measurements, save the example
    result = draw_lane(img, binaryWarped, leftFit, rightFit, src, dst)
    result = draw_measurements(result, (leftRadius + rightRadius) / 2, offset)
    cv2.imwrite('output_images/example_output_test3.jpg', result)

    # Display the final result
    cv2.imshow('lane on the original image', cv2.resize(result, None, fx=0.5, fy=0.5))

    # Wait until the window is closed before moving to the next step
    while cv2.getWindowProperty('lane on the original image', cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(100)

    cv2.destroyAllWindows()

    # Run the whole pipeline on the project videos (ESC stops the current video)
    os.makedirs('output_videos', exist_ok=True)

    for name in ('project_video01', 'project_video02', 'project_video03'):
        totalFrames, validFrames, slidingWindowFrames = process_video(
            'input_images/test_videos/' + name + '.mp4', 'output_videos/' + name + '.mp4', mtx, dist)

        print('{}: {} frames, {} valid ({:.1f}%), sliding windows used in {} frames'.format(
            name, totalFrames, validFrames, 100 * validFrames / totalFrames, slidingWindowFrames))


if __name__ == '__main__':
    main()

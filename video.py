import numpy as np
import cv2

from calibration import undistort
from thresholding import threshold_image
from perspective import get_perspective_points, warper
from lane_detection import find_lane_pixels, search_around_poly, fit_lines, is_lane_valid
from curvature import measure_curvature, measure_vehicle_offset
from drawing import draw_lane, draw_measurements

# Number of last valid frames used to smooth the polynomials
nframes = 5

# Number of frames in a row without a valid detection, after which the sliding windows are used again
maxLostFrames = 5


def process_video(inputPath, outputPath, mtx, dist, show=True):
    cap = cv2.VideoCapture(inputPath)

    # Video writer with the same size and frame rate as the input video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Polynomials of the last valid frames and the number of frames in a row without a valid detection
    leftFits = []
    rightFits = []
    lostFrames = maxLostFrames

    # Statistics for the writeup
    totalFrames = 0
    validFrames = 0
    slidingWindowFrames = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        totalFrames += 1

        # The calibration is valid only for the 1280x720 camera
        if frame.shape[1] == 1280:
            frame = undistort(frame, mtx, dist)

        # Thresholded binary image in the bird's-eye view
        src, dst = get_perspective_points(frame)
        binaryWarped = warper(threshold_image(frame), src, dst)

        # Search around the smoothed polynomials from the previous frames,
        # or start again with the sliding windows if the lines were lost
        if lostFrames < maxLostFrames:
            leftx, lefty, rightx, righty = search_around_poly(binaryWarped, leftFit, rightFit)
        else:
            leftx, lefty, rightx, righty, outImg = find_lane_pixels(binaryWarped)
            slidingWindowFrames += 1

        newLeftFit, newRightFit = fit_lines(leftx, lefty, rightx, righty)

        # Keep the new polynomials only if they pass the sanity check
        if newLeftFit is not None and is_lane_valid(newLeftFit, newRightFit, binaryWarped.shape):
            leftFits.append(newLeftFit)
            rightFits.append(newRightFit)
            validFrames += 1
            lostFrames = 0

            if len(leftFits) > nframes:
                leftFits.pop(0)
                rightFits.pop(0)
        else:
            lostFrames += 1

        result = frame

        # Draw the lane using the mean of the last valid polynomials
        if len(leftFits) > 0:
            leftFit = np.mean(leftFits, axis=0)
            rightFit = np.mean(rightFits, axis=0)

            leftRadius, rightRadius = measure_curvature(leftFit, rightFit, binaryWarped.shape)
            offset = measure_vehicle_offset(leftFit, rightFit, binaryWarped.shape)

            result = draw_lane(frame, binaryWarped, leftFit, rightFit, src, dst)
            result = draw_measurements(result, (leftRadius + rightRadius) / 2, offset)

        out.write(result)

        # Display the result, ESC stops the processing
        if show:
            cv2.imshow('lane finding', result)
            if (cv2.waitKey(1) & 0xff) == 27: break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return totalFrames, validFrames, slidingWindowFrames

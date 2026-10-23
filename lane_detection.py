import numpy as np
import cv2

# Sliding windows: number of windows, half width of a window (for 1280 px wide image)
# and the minimum number of pixels found to recenter the next window
nwindows = 9
margin = 100
minpix = 50


def get_histogram(binary):
    # Sum of the white pixels in each column of the bottom half of the image
    # The two highest peaks are the positions of the lane lines at the bottom of the image
    bottomHalf = binary[binary.shape[0] // 2:, :]
    return np.sum(bottomHalf // 255, axis=0)


def find_lane_pixels(binary):
    height, width = binary.shape
    histogram = get_histogram(binary)

    # Starting points: the highest peak in the left and in the right half of the histogram
    midpoint = width // 2
    leftxCurrent = np.argmax(histogram[:midpoint])
    rightxCurrent = np.argmax(histogram[midpoint:]) + midpoint

    # Scale the window width for the smaller (960x540) images
    windowMargin = int(margin * width / 1280)
    windowHeight = height // nwindows

    # Coordinates of all white pixels in the image
    nonzeroy, nonzerox = binary.nonzero()

    leftIndices = []
    rightIndices = []

    # Image for the visualization: binary image in color, windows are drawn in green
    outImg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Go through the windows from the bottom to the top of the image
    for window in range(nwindows):
        yLow = height - (window + 1) * windowHeight
        yHigh = height - window * windowHeight

        leftxLow = leftxCurrent - windowMargin
        leftxHigh = leftxCurrent + windowMargin
        rightxLow = rightxCurrent - windowMargin
        rightxHigh = rightxCurrent + windowMargin

        cv2.rectangle(outImg, (leftxLow, yLow), (leftxHigh, yHigh), (0, 255, 0), 2)
        cv2.rectangle(outImg, (rightxLow, yLow), (rightxHigh, yHigh), (0, 255, 0), 2)

        # Indices of the white pixels inside the windows
        inY = (nonzeroy >= yLow) & (nonzeroy < yHigh)
        goodLeft = (inY & (nonzerox >= leftxLow) & (nonzerox < leftxHigh)).nonzero()[0]
        goodRight = (inY & (nonzerox >= rightxLow) & (nonzerox < rightxHigh)).nonzero()[0]

        leftIndices.append(goodLeft)
        rightIndices.append(goodRight)

        # If enough pixels were found, recenter the next window on their mean position
        if len(goodLeft) > minpix:
            leftxCurrent = int(np.mean(nonzerox[goodLeft]))
        if len(goodRight) > minpix:
            rightxCurrent = int(np.mean(nonzerox[goodRight]))

    leftIndices = np.concatenate(leftIndices)
    rightIndices = np.concatenate(rightIndices)

    # Left line pixels in red, right line pixels in blue
    outImg[nonzeroy[leftIndices], nonzerox[leftIndices]] = (0, 0, 255)
    outImg[nonzeroy[rightIndices], nonzerox[rightIndices]] = (255, 0, 0)

    leftx, lefty = nonzerox[leftIndices], nonzeroy[leftIndices]
    rightx, righty = nonzerox[rightIndices], nonzeroy[rightIndices]

    return leftx, lefty, rightx, righty, outImg


def fit_polynomial(binary):
    leftx, lefty, rightx, righty, outImg = find_lane_pixels(binary)

    # Fit a second order polynomial x = A*y^2 + B*y + C to each line
    # x is a function of y, because the lines are almost vertical in the warped image
    leftFit = np.polyfit(lefty, leftx, 2)
    rightFit = np.polyfit(righty, rightx, 2)

    # Draw the polynomials in yellow
    ploty = np.arange(binary.shape[0])
    for fit in (leftFit, rightFit):
        fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
        points = np.int32(np.column_stack((fitx, ploty)))
        cv2.polylines(outImg, [points], False, (0, 255, 255), 3)

    return leftFit, rightFit, outImg

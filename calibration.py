import numpy as np
import cv2
import glob
import os

# Paths are relative to main.py
CALIB_IMAGES = 'data/camera_cal/calibration*.jpg'
CALIB_FILE = 'data/camera_cal/calib.npz'

# Define the chess board rows and columns
rows = 9
cols = 6

# Set the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)


def calibrate_camera(show=False):
    # Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0). They are the same for all images
    objectPoints = np.zeros((rows * cols, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # Create the arrays to store the object points and the image points
    objectPointsArray = []
    imgPointsArray = []
    imgSize = None

    # Loop over the image files
    for path in sorted(glob.glob(CALIB_IMAGES)):
        # Load the image and convert it to gray scale
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgSize = gray.shape[::-1]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

        # Make sure the chess board pattern was found in the image
        if ret:
            # Refine the corner position
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Add the object points and the image points to the arrays
            objectPointsArray.append(objectPoints)
            imgPointsArray.append(corners)

            # Draw the corners on the image
            cv2.drawChessboardCorners(img, (rows, cols), corners, ret)
        else:
            print('Chess board not found in: ', os.path.basename(path))

        # Display the image
        if show:
            cv2.imshow('chess board', img)

            # Wait until the window is closed before moving to the next image
            while cv2.getWindowProperty('chess board', cv2.WND_PROP_VISIBLE) >= 1:
                cv2.waitKey(100)

    if show:
        cv2.destroyAllWindows()

    print('Chess board found in {} images'.format(len(objectPointsArray)))

    # Calibrate the camera and save the results
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, imgSize, None, None)
    np.savez(CALIB_FILE, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    # Print the camera calibration error
    error = 0

    for i in range(len(objectPointsArray)):
        imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
        # Reshape both to (N, 2)
        error += cv2.norm(imgPointsArray[i].reshape(-1, 2), imgPoints.reshape(-1, 2), cv2.NORM_L2) / len(imgPoints)

    print('Total error: ', error / len(objectPointsArray))

    return mtx, dist


def load_calibration():
    # Load calibrated camera parameters
    if not os.path.exists(CALIB_FILE):
        return calibrate_camera()

    calibration = np.load(CALIB_FILE)
    return calibration['mtx'], calibration['dist']


def undistort(img, mtx, dist):
    # Undistort with the original camera matrix, so the image keeps the same size and scale
    return cv2.undistort(img, mtx, dist, None, mtx)

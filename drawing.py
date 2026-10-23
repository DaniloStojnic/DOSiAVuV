import numpy as np
import cv2

from perspective import warper


def draw_lane(img, binaryWarped, leftFit, rightFit, src, dst):
    height = binaryWarped.shape[0]

    # x positions of both lines for every row of the warped image
    ploty = np.arange(height)
    leftx = np.polyval(leftFit, ploty)
    rightx = np.polyval(rightFit, ploty)

    # Polygon of the lane
    leftPoints = np.column_stack((leftx, ploty))
    rightPoints = np.flipud(np.column_stack((rightx, ploty)))
    lanePolygon = np.int32(np.vstack((leftPoints, rightPoints)))

    # Draw the lane in green and the lines in red and blue on a blank warped image
    laneWarped = np.zeros_like(img)
    cv2.fillPoly(laneWarped, [lanePolygon], (0, 255, 0))
    cv2.polylines(laneWarped, [np.int32(leftPoints)], False, (0, 0, 255), 20)
    cv2.polylines(laneWarped, [np.int32(rightPoints)], False, (255, 0, 0), 20)

    # Warp the lane back to the original image (inverse perspective: dst -> src)
    laneUnwarped = warper(laneWarped, dst, src)

    # Combine the lane with the original image
    return cv2.addWeighted(img, 1, laneUnwarped, 0.3, 0)


def draw_measurements(img, radius, offset):
    # Write the radius of curvature and the vehicle position on the image
    side = 'right' if offset > 0 else 'left'
    radiusText = 'Radius of Curvature = {:.0f} m'.format(radius)
    offsetText = 'Vehicle is {:.2f} m {} of center'.format(abs(offset), side)

    scale = img.shape[1] / 1280
    for i, text in enumerate((radiusText, offsetText)):
        position = (int(30 * scale), int((50 + i * 50) * scale))
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (255, 255, 255), max(1, int(3 * scale)))

    return img

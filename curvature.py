import numpy as np

# Meters per pixel in the warped image (y direction, x direction) for each image size
METERS_PER_PIXEL = {
    (1280, 720): (14.6 / 360, 3.7 / 640),
    (960, 540): (14.6 / 290, 3.7 / 480),
}


def measure_curvature(leftFit, rightFit, imgShape):
    height, width = imgShape[:2]
    ymPerPix, xmPerPix = METERS_PER_PIXEL[(width, height)]

    # Convert the polynomial x = A*y^2 + B*y + C from pixels to meters:
    # x[m] = xm * x[px] and y[px] = y[m] / ym, so A[m] = A * xm / ym^2 and B[m] = B * xm / ym
    yEval = (height - 1) * ymPerPix

    radii = []
    for fit in (leftFit, rightFit):
        A = fit[0] * xmPerPix / ymPerPix ** 2
        B = fit[1] * xmPerPix / ymPerPix

        # Radius of curvature: R = (1 + (2Ay + B)^2)^(3/2) / |2A|
        radii.append((1 + (2 * A * yEval + B) ** 2) ** 1.5 / np.abs(2 * A))

    leftRadius, rightRadius = radii
    return leftRadius, rightRadius


def measure_vehicle_offset(leftFit, rightFit, imgShape):
    height, width = imgShape[:2]
    ymPerPix, xmPerPix = METERS_PER_PIXEL[(width, height)]

    # Position of the lines at the bottom of the image
    yBottom = height - 1
    leftx = np.polyval(leftFit, yBottom)
    rightx = np.polyval(rightFit, yBottom)

    # Positive offset: the car is right of the lane center, negative: left of the lane center
    laneCenter = (leftx + rightx) / 2
    offset = (width / 2 - laneCenter) * xmPerPix

    return offset

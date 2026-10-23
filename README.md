# Lane Finding Project

Goals and steps of the project:

* Compute the camera matrix and distortion coefficients from a set of chessboard images.
* Remove the distortion from road images.
* Create a binary image using color transforms and gradients.
* Apply a perspective transform to the binary image ("bird's-eye view").
* Find the lane line pixels and fit a polynomial through them.
* Compute the radius of curvature of the lane and the vehicle position with respect to the lane center.
* Warp the detected lane back onto the original image.
* Display the lane and the numerical values of the curvature and the vehicle position on the image and the video.

[//]: # (Image References)

[image1]: ./output_images/undistort_calibration1.jpg "Undistorted"
[image2]: ./output_images/undistort_test1.jpg "Road Transformed"
[image3]: ./output_images/binary_test1.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines1.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines_test3.jpg "Fit Visual"
[image6]: ./output_images/example_output_test3.jpg "Output"
[video1]: ./output_videos/project_video03.mp4 "Video"

---

## Project structure and running

| File | Content |
|---|---|
| [main.py](main.py) | Runs the image and video processing |
| [calibration.py](calibration.py) | Camera calibration and distortion removal |
| [thresholding.py](thresholding.py) | Creating the binary image |
| [perspective.py](perspective.py) | Perspective transform |
| [lane_detection.py](lane_detection.py) | Histogram, sliding windows, polynomial, search around the previous polynomial, result check |
| [curvature.py](curvature.py) | Radius of curvature and vehicle position |
| [drawing.py](drawing.py) | Drawing the lane and the values on the original image |
| [video.py](video.py) | Frame by frame video processing |

Input images and videos are in `input_images/`, the results of the individual steps in `output_images/`, and the processed videos in `output_videos/`.

The program is run from the main project folder:

```
python main.py
```

Each step shows its result in a window, and the next step starts when the window is closed. Video processing is stopped with the ESC key.

Used: Python 3.13, OpenCV 5.0, NumPy 2.5, Matplotlib 3.11.

---

## Camera Calibration

### 1. How the camera matrix and distortion coefficients were computed

**Functions** ([calibration.py](calibration.py)):

| Function | Responsibility |
|---|---|
| `calibrate_camera()` | Finds the chessboard corners on all images, computes the camera matrix and the distortion coefficients |
| `load_calibration()` | Loads the saved calibration parameters |
| `undistort()` | Removes the distortion from an image |

**Procedure**:

1. The inner corners of the chessboard are searched for on every image from `input_images/camera_cal/`. The chessboard has 9×6 inner corners.
2. "Object points" are the coordinates of the corners in the chessboard plane: (0,0,0), (1,0,0), …, (8,5,0). They are the same for all images, because the chessboard is flat (z = 0).
3. From the pairs (object points, image points) the camera matrix and the distortion coefficients are computed and saved in `calib.npz`.
4. As a check, the 3D chessboard points are projected back onto the image with the computed parameters and compared with the found corners.

**Results**:

* The chessboard was found on 17 of 20 images. On `calibration1`, `calibration4` and `calibration5` part of the chessboard is out of the frame, so it is not possible to find all 9×6 corners.
* The mean reprojection error is 0.12 px.

Example image before and after distortion removal:

![alt text][image1]

---

## Pipeline

### 1. Distortion removal

**Functions:** `undistort()` from [calibration.py](calibration.py).

Example image before and after distortion removal:

![alt text][image2]

**Problem**:

The dataset contains images from two different cameras. Calibration images exist only for the 1280×720 camera, so the calibration is used only for the images and videos of that camera. Videos with a resolution of 960×540 are processed without this step.

### 2. Binary image (color transforms and gradient)

**Functions** ([thresholding.py](thresholding.py)):

| Function | Responsibility |
|---|---|
| `threshold_image()` | Creates a binary image (0 and 255) from a color image |

**Procedure**:

1. The input is a color image with the distortion removed. It is converted to grayscale (for the gradient) and to HSV (for the yellow color).
2. Three masks are computed on the image, described in the table below.
3. The masks are combined into one. The output is a binary image with only two values: 255 for pixels that can belong to a lane line and 0 for everything else.
4. The example is saved as `binary_test1.jpg`.

| Mask | How it is computed |
|---|---|
| Gradient | Sobel in X on the grayscale image (3×3 kernel), threshold 80 |
| Yellow | Range in the HSV space: H 15–35, S 100–255, V 100–255 |
| White | Range in the BGR space: all three channels 200–255 |

Example of the undistorted and the binary image:

![alt text][image3]

**How the parameters were chosen**:

*Gradient (Sobel X):*

* **Threshold:** thresholds 30, 50, 80 and 120 were tried. With lower thresholds the mask picks up the asphalt texture and shadows, and with threshold 120 the distant and dashed parts of the lines are lost. Threshold 80 was chosen.
* **Kernel size:** The 3×3 kernel turned out better than 5×5 because it gave fewer false edges.
* **Direction:** Sobel X turned out better than Sobel Y and the gradient magnitude because the lane lines are mostly vertical or diagonal. Sobel Y picked up many horizontal edges, such as shadows and changes of the road surface.

*Yellow mask (lower bound for S and V):*

* **≥ 50:** the mask picks up trees, dry grass and hills.
* **≥ 100 (chosen):** the line is complete and there is little noise.
* **≥ 120:** the cleanest, but up to 40% of the line is lost.

*White mask.*

* Using only the R channel of the RGB space was tried. For `R > 200`, the mask also picks up the yellow line and the bright part of the concrete.
* A better result was obtained by requiring all three BGR channels to be greater than 200.

### 3. Perspective transform

**Functions** ([perspective.py](perspective.py)):

| Function | Responsibility |
|---|---|
| `get_perspective_points()` | Returns the source and destination points for the resolution of the given image |
| `warper()` | Computes the perspective transform matrix and applies it to the image |

**Procedure**:

1. The input is the binary image from the previous step.
2. A trapezoidal part of the road is mapped into a rectangle.
3. After the transform the road looks as if it is viewed from above, and the lane lines become approximately parallel.
4. The example is saved as `warped_straight_lines1.jpg`.

| Camera | Source (src) | Destination (dst) |
|---|---|---|
| 1280×720 | (582, 460), (703, 460), (1106, 720), (213, 720) | (320, 0), (960, 0), (960, 720), (320, 720) |
| 960×540 | (426, 345), (542, 345), (848, 540), (148, 540) | (240, 0), (720, 0), (720, 540), (240, 540) |

Example of the binary image before and after the transform:

![alt text][image4]

### 4. Finding the lane line pixels and the polynomial

**Functions** ([lane_detection.py](lane_detection.py)):

| Function | Responsibility |
|---|---|
| `get_histogram()` | Counts the white pixels in each column of the bottom half of the image (column histogram) |
| `find_lane_pixels()` | Determines the start of the lines from the histogram peaks and collects the pixels of the left and right line with sliding windows |
| `fit_polynomial()` | Fits a second order polynomial through the pixels of each line and draws it |

**Procedure**:

1. The input is the binary image in the bird's-eye view, from the previous step.
2. **Histogram:** for each column of the bottom half of the image, the number of white pixels is counted. The highest peak in the left and in the right half of the histogram are the starting positions of the left and right line. Only the bottom half is used because there the line is closest to the car and least shifted by the curve, and the upper part of the image has more noise.
3. **Sliding windows:** the image is divided into 9 parts by height. For each line a window starts at the bottom of the image and moves towards the top, collecting the white pixels inside it. If the window finds enough pixels, the next window above is moved to their mean position, so the windows follow the line even when it curves.
4. **Polynomial:** a second order polynomial `x = Ay² + By + C` is fitted through the collected pixels of each line. x is a function of y, because the lines in the bird's-eye view are almost vertical.
5. The output are the coefficients A, B and C for the left and right line. The example is saved as `color_fit_lines_test3.jpg`.

Example (`test3.jpg`): the windows are green, the left line pixels red, the right line pixels blue, and the polynomials yellow.

![alt text][image5]

The coefficients for this example are left `A = 1.26·10⁻⁴, B = −0.298, C = 481.7` and right `A = 1.82·10⁻⁴, B = −0.296, C = 1105.9` (in pixels). C is the position of the line at the top of the image, B its slope, and A shows how much the line is curved. Both lines have a positive A of similar size, so they curve to the same side, as they should in a curve.

### 5. Radius of curvature and vehicle position

**Functions** ([curvature.py](curvature.py)):

| Function | Responsibility |
|---|---|
| `measure_curvature()` | Converts the polynomial coefficients from pixels to meters and computes the radius of curvature of the left and right line |
| `measure_vehicle_offset()` | Computes the offset of the car center from the lane center in meters |

**Procedure**:

1. The input are the coefficients of the polynomial `x = Ay² + By + C` of the left and right line from the previous step, in pixels.
2. The coefficients are converted from pixels to meters, where `xm` is the number of meters per pixel along the width, and `ym` along the height of the image in the bird's-eye view:

   ```
   A[m] = A · xm / ym²        B[m] = B · xm / ym
   ```

3. The radius of curvature is computed with:

   ```
   R = (1 + (2Ay + B)²)^(3/2) / |2A|
   ```

   The value is computed at the bottom of the image, because that part of the road is closest to the car. The mean of the left and right line is used for display.

4. The vehicle position is computed by comparing the image center and the lane center at the bottom of the image:

   ```
   position = (width / 2 − (x_left + x_right) / 2) · xm
   ```

   A positive value means that the car is right of the lane center, and a negative value that it is left.

5. The output are the radius of curvature and the vehicle position in meters.

**How `xm` and `ym` were determined:**

* **`xm`:** A lane width of 3.7 m was used. In the bird's-eye view the distance between the lines for a 1280×720 image is approximately 640 px, so `xm = 3.7 / 640` m/px is used. For a 960×540 image the corresponding distance of approximately 480 px is used.
* **`ym`:** Estimated using the distance between the starts of neighboring dashed lines. For the 1280×720 resolution the distance is approximately 360 px, and for 960×540 about 290 px. A real distance of approximately 14.6 m was assumed.

Results on the test images:

| Image | R left | R right | Mean | Position |
|---|---:|---:|---:|---:|
| test1 | 566 m | 1 380 m | 973 m | 0.15 m left |
| test2 | 605 m | 1 019 m | 812 m | 0.36 m left |
| test3 | 1 126 m | 783 m | 955 m | 0.12 m left |
| test5 | 784 m | 865 m | 825 m | 0.02 m right |
| test6 | 1 138 m | 598 m | 868 m | 0.28 m left |
| straight_lines1 | 7 774 m | 6 778 m | 7 276 m | 0.01 m right |

On curved parts of the road the values are of the order of about 800–1000 m. On an almost straight road much larger values are obtained, thousands of meters, which is expected because a straight line has an infinite radius of curvature.

### 6. Example of the result warped back onto the original image

**Functions** ([drawing.py](drawing.py)):

| Function | Responsibility |
|---|---|
| `draw_lane()` | Draws the lane and the lines in the bird's-eye view, warps them back to the normal view and overlays them on the original image |
| `draw_measurements()` | Writes the radius of curvature and the vehicle position on the image |

**Procedure**:

1. The input are the image with the distortion removed, the polynomials of the left and right line from step 4, and the radius and vehicle position from step 5.
2. On a blank image in the bird's-eye view, the area between the two polynomials is filled with green, and the left line is drawn in red and the right line in blue.
3. The resulting image is warped back to the original perspective using the inverse perspective transform.
4. The warped-back lane image is then overlaid on the undistorted image with 30% opacity, so the road is still visible under it.
5. The radius of curvature and the vehicle position are written in the upper part of the image. The example is saved as `example_output_test3.jpg`.

Example (`test3.jpg`):

![alt text][image6]

---

## Pipeline (video)

### 1. Link to the processed video

* [project_video01.mp4](./output_videos/project_video01.mp4) (960×540)
* [project_video02.mp4](./output_videos/project_video02.mp4) (960×540)
* [project_video03.mp4](./output_videos/project_video03.mp4) (1280×720)

**Functions:**

| Function | Responsibility |
|---|---|
| `process_video()` ([video.py](video.py)) | Reads the video frame by frame, runs the whole pipeline for each frame, keeps the polynomials from the previous frames and writes the result into a new video |
| `search_around_poly()` ([lane_detection.py](lane_detection.py)) | Searches for the lane line pixels in a band around the polynomials from the previous frames |
| `fit_lines()` ([lane_detection.py](lane_detection.py)) | Fits the polynomials only if both lines have enough pixels |
| `is_lane_valid()` ([lane_detection.py](lane_detection.py)) | Checks whether the found lane has a reasonable width and whether the lines are approximately parallel |

**Procedure**: For the first frame the standard procedure with the histogram and sliding windows is used. For the following frames additional checks and information from the previous frames are used:

1. **Search around the previous polynomial:** Pixels are searched for in a band of ±100 px around the polynomials from the previous frames, instead of running the sliding windows again.
2. **Protection against too few pixels:** A new polynomial is accepted only if both lines have at least 500 found pixels. If the number is much smaller, it is very likely that noise or only a small part of the line was found.
3. **Result check:** The lane width is checked at the top, middle and bottom of the image in the bird's-eye view. An acceptable result must have a width in the range of 70–130% of the expected value. This check indirectly also checks that the left and right line are approximately parallel.
4. **Averaging:** The mean of the polynomials of the last 5 valid frames is used for drawing. This reduces the jitter of the lines, while keeping a fast enough reaction to changes of the road direction.
5. **Search again:** if there is no good result for 5 frames in a row, the sliding windows are started again.

**Result:** the lane is correctly marked along the whole length of all three videos, including the parts with bright concrete and tree shadows.

---

## Discussion

### 1. Problems, where the pipeline may fail and how to improve it

**Problems during the implementation:**

* **Two cameras.** The data was recorded with two cameras of different resolution, and calibration images exist only for one. Therefore the distortion is removed only from 1280×720 images, and the points for the perspective transform were determined separately for each resolution.
* **Bright concrete.** On parts of the road with bright concrete the yellow line has a small difference in brightness compared to the surface, so the gradient picks it up poorly. That line is found by the yellow color mask.
* **Radius on a straight road.** On a straight road the line is almost straight, so the coefficient A is close to zero. Because of that, a small amount of noise changes the computed radius a lot, so the displayed value on straight parts of the road jumps and can be very large. This does not mean that the lane was found incorrectly, but that a gentle curve and a straight road cannot be precisely distinguished.

**Where the pipeline will likely fail:**

* **Lane change:** the pipeline tracks only its own lane, so while the car crosses a line, there are no clear left and right lines.
* **Uphill, downhill and car rocking:** the perspective transform assumes a flat road and a steady camera. When the camera tilts, the lines in the bird's-eye view stop being parallel, so a false curvature appears.
* **Sharp curves:** the line leaves the trapezoid of the perspective transform, so part of the line pixels is lost.
* **Sudden change of lighting** (shadows, passing under a bridge, night): the thresholds for color and gradient are fixed, so they stop being valid.

**How to make the pipeline more robust:**

* Raise the bottom of the perspective transform trapezoid above the car hood (about y = 680 for 1280×720), because the lane is now drawn over the hood as well.
* Use adaptive thresholds instead of fixed ones, so that the binary image is more resistant to lighting changes.
* Also track the neighboring lanes, which would help during a lane change.

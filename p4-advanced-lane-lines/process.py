import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# Load camera matrices to later undistort images
pickle_data = pickle.load(open('./callibration_pickle', 'rb'))
mtx = pickle_data["mtx"]
dist = pickle_data["dist"]

ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/700  # meters per pixel in x dimension


def line_finder(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    y_eval = np.max(ploty)
    l_width = 10

    left_lane = np.array(list(zip(np.concatenate((left_fitx-l_width, left_fitx[::-1] + l_width), axis=0), np.concatenate((ploty, ploty[::-1]), axis=0))), dtype=np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-l_width, right_fitx[::-1] + l_width), axis=0), np.concatenate((ploty, ploty[::-1]), axis=0))), dtype=np.int32)
    inner_lane = np.array(list(zip(np.concatenate((left_fitx+l_width, right_fitx[::-1] - l_width), axis=0), np.concatenate((ploty, ploty[::-1]), axis=0))), dtype=np.int32)

    # Find camera position
    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - binary_warped.shape[1] / 2) * xm_per_pix

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_lane, inner_lane, right_lane, center_diff, left_curverad, right_curverad


# Utilities for working with different color spaces
# Use grayscale color space
def gray_select(img, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray >= thresh[0]) & (gray <= thresh[1])] = 1

    return binary


# Use a selected chanel of RGB color space
def rgb_select(img, thresh, chanel=0):
    img_ch = img[:, :, chanel]
    binary = np.zeros_like(img_ch)
    binary[(img_ch > thresh[0]) & (img_ch <= thresh[1])] = 1

    return binary


# Use a selected chanel of HLS color space
def hls_select(img, thresh, chanel=2):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_ch = hls[:, :, chanel]
    binary = np.zeros_like(img_ch)
    binary[(img_ch >= thresh[0]) & (img_ch <= thresh[1])] = 1

    return binary


# Use a selected chanel of HSV color space
def hsv_select(img, thresh, chanel=2):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_ch = hsv[:, :, chanel]
    binary = np.zeros_like(img_ch)
    binary[(img_ch >= thresh[0]) & (img_ch <= thresh[1])] = 1

    return binary


# Utilities for working with gradient
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)

    scale_factor = np.max(abs_sobelxy) / 255
    abs_sobelxy = (abs_sobelxy / scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(abs_sobelxy)
    binary_output[(abs_sobelxy >= mag_thresh[0]) & (abs_sobelxy <= mag_thresh[1])] = 1

    return binary_output


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255, sobel_kernel=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_64F, int(orient == 'x'), int(orient == 'y'), ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8((255 * abs_sobel / np.max(abs_sobel)))

    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return sbinary


# Performs binary AND over two arrays
def b_and(im1, im2):
    binary = np.zeros_like(im1)
    binary[(im1 == 1) & (im2 == 1)] = 1

    return binary


# Performs binary OR over two arrays
def b_or(im1, im2):
    binary = np.zeros_like(im1)
    binary[(im1 == 1) | (im2 == 1)] = 1

    return binary


def image_pipeline(image):
    # get undistorted frame
    image = cv2.undistort(image, mtx, dist, None, mtx)

    img_size = (image.shape[1], image.shape[0])
    offset = img_size[0] * .3

    # Compute gradients in X and Y directions
    gradx = abs_sobel_thresh(image, 'x', 25, 255, sobel_kernel=3)
    grady = abs_sobel_thresh(image, 'y', 10, 255, sobel_kernel=3)

    # Combine X and Y gradients using binary AND
    grad = b_and(gradx, grady)

    # Use median blur filter to remove small noise
    grad_cleaned = cv2.medianBlur(grad, 5)

    # Get the S chanel in HLS color space
    hls_s_binary = hls_select(image, thresh=(100, 255), chanel=2)

    # Get the V chanel in HSV color space
    hsv_v_binary = hsv_select(image, thresh=(100, 255), chanel=2)

    # Combine S and V channels using binary AND
    hls_s_lsv_v = b_and(hls_s_binary, hsv_v_binary)

    # Combine result of color threshold with gradient threshold
    result = b_or(grad_cleaned, hls_s_lsv_v)


    # Define points for perspective transformation
    src = np.float32(([602, 446], [682, 446], [1027, 665], [294, 665]))
    dst = np.float32([[offset, 0], [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])

    # Compute matrices for warping and "unwarping"
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the frame
    warped = cv2.warpPerspective(result, M, img_size, flags=cv2.INTER_LINEAR)

    # Find the lines, position of the camera and the curvature
    left_lane, inner_lane, right_lane, center_diff, left_curverad, right_curverad = line_finder(warped)

    # Draw the lines
    road = np.zeros_like(image)
    cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road, [inner_lane], color=[0, 255, 0])

    # Get the image of the lines with perspective
    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)

    # Add image of the line to the frame
    result = cv2.addWeighted(image, 1.0, road_warped, 0.5, 0.0)

    if center_diff > 0:
        side_pos = 'left'
    else:
        side_pos = 'right'

    # Add additional information
    cv2.putText(result, 'Radius of Curvature = ' + str(round(left_curverad, 3)) + 'm', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    return result

input_video = 'project_video.mp4'
output_video = 'project_video_output.mp4'

clip = VideoFileClip(input_video)
video_clip = clip.fl_image(image_pipeline)
video_clip.write_videofile(output_video, audio=False)

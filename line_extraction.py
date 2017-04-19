import numpy as np
import cv2
import matplotlib.pyplot as plt


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # radius history
        self.radiuses = np.array([])

    def append_curv(self,curv_radius):
        AVG_SIZE = 10
        self.radiuses = np.append(self.radiuses,curv_radius)
        size = self.radiuses.shape[0]
        if size > AVG_SIZE:
            self.radius_of_curvature = np.mean(self.radiuses[-AVG_SIZE:])
        else:
            self.radius_of_curvature = np.mean(self.radiuses[-size:])

def find_lines(binary_warped,left_line=None,right_line=None,axis=None):

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 500

    if not left_line.detected or not right_line.detected:
        left_lane_inds, right_lane_inds,out_img = find_lines_from_scratch(binary_warped, right_line, left_line,
                                                                          nonzerox, nonzeroy, margin, minpix)
    else:
        # reuse previous information
        left_fit = left_line.current_fit
        right_fit = right_line.current_fit
        # Search for pixels along the previous fit directly
        left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Check that the detected line has enough pixels to be trusted
    cnt_left = len(leftx)
    cnt_right = len(rightx)
    bot_y = binary_warped.shape[0]
    MIN_PIXEL_CNT = 4300
    # Use strong line info as replacement for a weak line
    if cnt_left > MIN_PIXEL_CNT and cnt_right < MIN_PIXEL_CNT:
        correct_fit_intersection(bot_y, right_fit,left_fit)
    if cnt_left < MIN_PIXEL_CNT and cnt_right > MIN_PIXEL_CNT:
        correct_fit_intersection(bot_y, left_fit, right_fit)

    if axis is not None:
        plot_line_fit_area(axis, binary_warped, left_fit, right_fit,
                                                leftx,lefty,
                                                rightx,righty,
                                                margin)
        res_img = out_img
    else:
        left_fitx, right_fitx, ploty, res_img = get_line_fit_image(binary_warped, left_fit, right_fit,
                                                                   leftx,lefty,
                                                                   rightx,righty,
                                                                   margin)
        cv2.putText(res_img, "Pix Cnt L " + str(len(leftx)),
                    (400, 300), cv2.FONT_HERSHEY_SIMPLEX, thickness=3, fontScale=1, color=[255, 255, 255])
        cv2.putText(res_img, "Pix Cnt R " + str(len(rightx)),
                    (500, 500), cv2.FONT_HERSHEY_SIMPLEX, thickness=3, fontScale=1, color=[255, 255, 255])


    curv_left, curv_right,offset_m = calc_curvature(leftx,lefty,rightx,righty,binary_warped)
    left_line.append_curv(curv_left)
    left_line.current_fit = left_fit

    right_line.append_curv(curv_right)
    right_line.current_fit = right_fit

    return left_line,right_line,res_img,offset_m


def correct_fit_intersection(bot_y, incorrect_fit, ref_fit):
    # Compute last coefficient using the start point of the weak line but
    # applying strong fit to correct the intersection term C:
    # leftx_base = A*Y**2 + B*Y + C
    # C_corected = leftx_base - (A*Y**2 + B*Y)
    leftx_base = incorrect_fit[0] * (bot_y ** 2) + incorrect_fit[1] * bot_y + incorrect_fit[2]
    incorrect_fit[2] = leftx_base - (ref_fit[0] * bot_y ** 2 + ref_fit[1] * bot_y)
    # Copy the curvature information from strong line
    incorrect_fit[0] = ref_fit[0]
    incorrect_fit[1] = ref_fit[1]


def find_lines_from_scratch(binary_warped, right_line, left_line, nonzerox, nonzeroy, margin,minpix):
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
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
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
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
    left_line.detected = True
    right_line.detected = True
    return left_lane_inds,right_lane_inds,out_img


def calc_curvature(leftx,lefty,rightx,righty,binary_warped):

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension®
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')

    # Base position:
    base_x_left_m = left_fit_cr[0]*(y_eval*ym_per_pix)**2 + left_fit_cr[1]*(y_eval*ym_per_pix) + left_fit_cr[2]
    base_x_right_m = right_fit_cr[0]*(y_eval*ym_per_pix)**2 + right_fit_cr[1]*(y_eval*ym_per_pix) + right_fit_cr[2]
    line_center = (base_x_right_m + base_x_left_m) / 2
    image_center = xm_per_pix*binary_warped.shape[1]/2
    offset_m = abs(line_center - image_center)

    return left_curverad,right_curverad,offset_m


def plot_line_fit(axis,binary_warped,left_fit,right_fit,nonzerox,nonzeroy,
                  left_lane_inds,right_lane_inds,out_img):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    axis.imshow(out_img)
    axis.plot(left_fitx, ploty, color='yellow')
    axis.plot(right_fitx, ploty, color='yellow')
    #axis.xlim(0, 1280)
    #axis.ylim(720, 0)

def plot_line_fit_area(axis, binary_warped, left_fit, right_fit,
                                                leftx,lefty,
                                                rightx,righty,
                                                margin):
    left_fitx, right_fitx, ploty, result  = get_line_fit_image(axis, binary_warped, left_fit, right_fit,
                                                leftx,lefty,
                                                rightx,righty,
                                                margin)
    axis.imshow(result)
    axis.plot(left_fitx, ploty, color='yellow')
    axis.plot(right_fitx, ploty, color='yellow')
    axis.set_axis_off()
    axis.set_title("Poly Fit")


def get_line_fit_image(binary_warped, left_fit,right_fit,
                       leftx,lefty,
                       rightx,righty,
                       margin):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return left_fitx, right_fitx,ploty, result
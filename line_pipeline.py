from calibration import *
from thresholding import *
from line_extraction import *


def undistort(img,calib):
    """
    Distortion correction
    :param img: input image
    :param calib: calibration matrix
    :return: undistorded image
    """
    undist = cv2.undistort(img, calib.mtx, calib.dist, None, calib.mtx)
    return undist

# Color/gradient threshold
def apply_thresholds(img_rgb):

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gradx = abs_sobel_thresh(gray, 1,0, thresh=(20, 255))
    mag_binary = mag_thresh(img_rgb, mag_thresh=(20, 255))
    binary, h_chan, l_chan, s_chan = s_channel_threshold(img_rgb)

    # Combine all these together:
    final = np.zeros_like(mag_binary)
    final[(binary == 1) | ((gradx == 1) & (mag_binary == 1)) ] = 1

    return final,l_chan,s_chan

# Perspective transform
def transform_perspective(img,draw_lines=False):
    length, width = img.shape[1], img.shape[0]

    # Define shape for perspective transformation
    lb = (130,width)
    rb = (1235,width)
    lt = (560,465)
    rt = (730,465)

    selection_img = np.copy(img)
    drawLinesFromPoints(lb, lt, rt, rb, selection_img)

    offset = 200
    src = np.float32([lb, lt, rt, rb])
    d_lb = (offset,width)
    d_rb = (offset, 0)
    d_lt = (length - offset, 0)
    d_rt = (length - offset, width)
    dst = np.float32([d_lb, d_rb, d_lt, d_rt])

    # get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    # Perspective transform
    warped = cv2.warpPerspective(img, M, (length,width), flags=cv2.INTER_LINEAR)

    if draw_lines:
        drawLinesFromPoints(d_lb,d_rb,d_lt,d_rt,warped)

    return warped, selection_img, Minv


def drawLinesFromPoints(p1, p2, p3, p4, img):
    cv2.line(img, p1, p2, color=[0, 255, 0], thickness=5)
    cv2.line(img, p2, p3, color=[0, 255, 0], thickness=5)
    cv2.line(img, p3, p4, color=[0, 255, 0], thickness=5)
    cv2.line(img, p4, p1, color=[0, 255, 0], thickness=5)

# Draw resulting lines back onto the image
def draw_result(warped, left_fit, right_fit, mtx, undist):
    # Compute points from lift fits
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, mtx, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

class LinePipeline():
    def __init__(self,debugView=True,usePrevLines=True,undistort=False):
        self.left_line = Line()
        self.right_line = Line()
        self.frame_nb = 0
        self.undistort = undistort
        if undistort:
            self.calibration = Calibration()
            self.calibration.load()
        self.debugView = debugView
        self.usePrevLines = usePrevLines

    def process(self,img_rgb):
        # Undistort image with calibration data
        if self.undistort:
            undistorted = undistort(img_rgb,self.calibration)
        else:
            undistorted = img_rgb
        # Threshold and wrap
        thresholded,l_chan,s_chan = apply_thresholds(undistorted)
        warped_thresh, ignr,Minv = transform_perspective(thresholded)

        # Find lines and draw result to the image
        self.left_line, self.right_line,res_img,offset_m = find_lines(warped_thresh,
                                                                        self.left_line,
                                                                        self.right_line)
        result = draw_result(warped_thresh, self.left_line.current_fit,
                             self.right_line.current_fit, Minv, undistorted)

        # Write curvature info on image
        avg_curv = (self.left_line.radius_of_curvature + self.right_line.radius_of_curvature)/2
        cv2.putText(result, "Curv. Radius: " + str(avg_curv) + " Lane Offset:" + str(offset_m) ,
                    (200, 100), cv2.FONT_HERSHEY_SIMPLEX, thickness=3,fontScale=1,color=[0,0,0])

        # Extra debug mode
        if self.debugView:
            cv2.putText(res_img, "Frame Nb: " + str(self.frame_nb),
                        (400, 100), cv2.FONT_HERSHEY_SIMPLEX, thickness=3, fontScale=1, color=[255, 255, 255])
            vis = np.concatenate((res_img, result), axis=0)
            self.frame_nb += 1
        else:
            vis = result

        if not self.usePrevLines:
            self.left_line = None
            self.right_line = None

        return vis

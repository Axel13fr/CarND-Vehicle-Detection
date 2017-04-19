import numpy as np
import cv2

# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient_x,orient_y, sobel_kernel=3,thresh=(0,255)):
    # 1) Assumes the image is one dimension in depth !
    assert(len(img.shape) == 2)
    # 2) Take the derivative based on orientation vector
    sobel = cv2.Sobel(img, cv2.CV_64F, orient_x,orient_y, ksize=sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    soble_abs = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    soble_norm = np.uint8((255 * soble_abs / np.max(soble_abs)))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(soble_norm)
    binary_output[(soble_norm > thresh[0]) & (soble_norm < thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

# Define a function that applies Sobel x or y,
# then takes the magnitude value and applies a threshold.
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 3) Take the absolute value of the derivative or gradient
    soble_abs = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    soble_norm = np.uint8((255 * soble_abs / np.max(soble_abs)))
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(soble_norm)
    mag_binary[(soble_norm > mag_thresh[0]) & (soble_norm < mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    # Apply threshold
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Apply threshold
    return binary_output


def s_channel_threshold(rgb_img,thresh=(180, 255)):
    """
    Applies threshold to the S channel of the HLS image converted from RGB input
    :param rgb_img: an RGB image
    :param thresh: the thresholds to apply
    :return: the binary image output
    """
    h_chan, l_chan, s_chan =cv2.split(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS))
    binary = np.zeros_like(s_chan)
    binary[(s_chan > thresh[0]) & (s_chan <= thresh[1])] = 1
    return binary,h_chan,l_chan,s_chan


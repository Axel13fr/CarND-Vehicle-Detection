import numpy as np
import cv2
import matplotlib.image as mpimg
from skimage.feature import hog

class ExtractSettings():
    ''' Container for feature extraction parameters '''

    def __init__(self):
        # Color Space: applies for bin_spatial and hog features only
        self.cspace = 'YUV'  # HSV with all gives 0.97, HLS 0.96, YUV 0.98
        self.spatial = 15

        # Color histogram bin setting. Done on HSV channels, independant of cspace setting
        self.histbin = 30

        # HOG Settings
        # HSV : 0 gives 0.93, 1 gives 0.88, 2 gives 0.94, ALL gives 0,97
        # YUV : 0 gives 0.94, 1 gives 0.94, 2 gives 0.90, ALL gives 0,98
        # YU from YUV: 0.97
        self.hog_channel = 'SELECT'
        self.selected_chans = [0,1]
        self.orient = 6 # 9 gives 0.93, 6 gives 0.92
        self.pix_per_cell = 12 # 8 gives 0,93, 10 gives 0.92, 12 gives 0.92
        self.cell_per_block = 1 # 1 gives 0.93, 2 gives 0.94

    def print(self):
        print('Using spatial binning of:', self.spatial,
              'and', self.histbin, 'histogram bins')
        print('Color space is ',self.cspace,
              'and hog channel uses ',self.hog_channel)

def convert_color(img, color_space):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)

    return feature_image

# Define a function to compute color histogram features
def color_hist(img_hsv, nbins=32, bins_range=(0, 256)):
    # Optimal color space for color hist is HSV
    #img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img_hsv[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img_hsv[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img_hsv[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    feature_image = convert_color(img,color_space)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    #if vis == True:
    #    features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
    #                              cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
    #                              visualise=True, feature_vector=False)
    #    return features, hog_image
    #else:
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
    return features


def single_img_features(img_rgb,settings,spatial_feat=True,
                                       hist_feat=True, hog_feat=True):
    spatial_size = (settings.spatial,settings.spatial)
    hist_bins = settings.histbin

    # 1) Define an empty list to receive features
    img_features = []
    # 2) Convert image to new color space (if specified)
    feature_image = convert_color(img_rgb,settings.cspace)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(img_rgb, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if settings.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     settings.orient, settings.pix_per_cell,
                                                     settings.cell_per_block,
                                                     vis=False, feature_vec=True))
        elif settings.hog_channel == 'SELECT':
            hog_features = []
            for channel in settings.selected_chans:
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     settings.orient, settings.pix_per_cell,
                                                     settings.cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, settings.hog_channel],
                                            settings.orient, settings.pix_per_cell,
                                            settings.cell_per_block,
                                            vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    return np.concatenate(img_features)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs,settings,
                     spatial_feat=True, hist_feat=True, hog_feat=True):

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        img_rgb = mpimg.imread(file)
        img_features = single_img_features(img_rgb,settings,
                                           spatial_feat,hist_feat,hog_feat)
        # Append the new feature vector to the features list
        features.append(img_features)

    # Return list of feature vectors
    return features


def combine_normalize(car_features,notcar_features):
    # Create an array stack, NOTE: StandardScaler() expects np.float64
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    from sklearn.preprocessing import StandardScaler
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    return scaled_X,y,X_scaler


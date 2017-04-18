import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from feature_extraction import *
from train import *
from sklearn.model_selection import train_test_split


# Read in car and non-car images
cars,notcars = read_images("train_images")
data_look(cars,notcars)

# TODO play with these values to see how your classifier
# performs under different binning scenarios
spatial = 10
histbin = 10
cspace = 'HSV'
hog_channel=0


notcar_features = extract_features(notcars, cspace=cspace, spatial_size=(spatial, spatial),
                        hist_bins=histbin, hist_range=(0, 256))
car_features = extract_features(cars, cspace=cspace, spatial_size=(spatial, spatial),
                        hist_bins=histbin, hist_range=(0, 256))

scaled_X,y = combine_normalize(car_features,notcar_features)
print("Scaled feature vector size: " + str(scaled_X.shape))
print('Using spatial binning of:',spatial,
    'and', histbin,'histogram bins')
print('Feature vector length:', len(car_features[0]))

print(" -------------------------------------------------------------------- ")
train(scaled_X,y)
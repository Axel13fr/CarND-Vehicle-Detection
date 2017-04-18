import time
import glob
import pickle
from feature_extraction import *

def read_images(folder):
    images = glob.glob(folder + '/*/*/*.png')
    cars = []
    notcars = []

    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)

    return cars,notcars

# Define a function to return some characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype

    print('data_look returned a count of',
          data_dict["n_cars"], ' cars and',
          data_dict["n_notcars"], ' non-cars')
    print('of size: ', data_dict["image_shape"], ' and data type:',
          data_dict["data_type"])

    # Return data_dict
    return data_dict

def generate_features():

    # Read in car and non-car images
    cars, notcars = read_images("train_images")
    data_look(cars, notcars)

    # Different Feature Extraction settings : see ExtractSettings in feature_extraction.py
    settings = ExtractSettings()

    notcar_features = extract_features(notcars, settings,
                                       spatial_feat=True, hist_feat=True, hog_feat=True)
    car_features = extract_features(cars, settings,
                                    spatial_feat=True, hist_feat=True, hog_feat=True)
    settings.print()

    # Save into pickle file for retraining !
    feat_pickle = {}
    feat_pickle["notcar_features"] = notcar_features
    feat_pickle["car_features"] = car_features
    feat_pickle['settings'] = settings
    pickle.dump(feat_pickle, open("features/features.p", "wb"))

    scaled_X, y = combine_normalize(car_features, notcar_features)
    print("Scaled feature vector size: " + str(scaled_X.shape))

    return scaled_X, y

def load_features():
    with open('features/features.p', 'rb') as handle:
        features = pickle.load(handle)
        notcar_features = features["notcar_features"]
        car_features = features["car_features"]
        settings = features["settings"]
        settings.print()

        scaled_X, y = combine_normalize(car_features, notcar_features)
        print("Scaled feature vector size: " + str(scaled_X.shape))
    return scaled_X,y


class Model():
    def __init__(self):
        self.model = None

    def load(self,pickle_file):
        with open('svm/model.p', 'rb') as handle:
            pick = pickle.load(handle)
            self.model = pick["model"]

    def train(self,scaled_X,y):

        from sklearn.model_selection import train_test_split
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        from sklearn import svm
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.model_selection import TimeSeriesSplit

        # Grid search over a set of parameters
        C_range = np.logspace(-5, -1, 10)
        gamma_range = np.logspace(-9, 3, 10)
        #parameters = {'kernel': ('linear', 'rbf'), 'C': C_range,'gamma'=gamma_range}
        parameters = {'C': C_range}
        # Check the training time for the SVC
        t = time.time()

        # Gridsearch with custom split
        #tscv = TimeSeriesSplit(n_splits=3)
        scv = StratifiedShuffleSplit(n_splits=5, test_size=0.2,random_state=rand_state)
        grid = GridSearchCV(svm.LinearSVC(), parameters, cv=scv)
        grid.fit(scaled_X, y)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train & search grid params')
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))

        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(grid.score(X_test, y_test), 4))

        self.model = grid

        # Save the trained model
        svm_pickle = {}
        svm_pickle['model'] = grid
        pickle.dump(svm_pickle, open("svm/model.p", "wb"))

    def predict(self,X):
        return self.model.predict(X)
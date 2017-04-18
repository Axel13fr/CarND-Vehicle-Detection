import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import numpy as np
import glob

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

def train(scaled_X,y):

    from sklearn.model_selection import train_test_split
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedShuffleSplit

    # Grid search over a set of parameters
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    #parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    parameters = {'C': C_range}
    # Check the training time for the SVC
    t = time.time()

    # Gridsearch will do a 3-fold cross validation by default
    grid = GridSearchCV(svm.LinearSVC(), parameters,cv=None)
    grid.fit(scaled_X, y)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train & search grid params')
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(grid.score(X_test, y_test), 4))


    return grid
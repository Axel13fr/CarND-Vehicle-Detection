import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob

def read_images(folder):
    images = glob.glob(folder + '/*.jpeg')
    cars = []
    notcars = []

    for image in images:
        if 'image' in image or 'extra' in image:
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

    from sklearn.svm import LinearSVC
    # Use a linear SVC (support vector classifier)
    svc = LinearSVC()
    # Train the SVC
    svc.fit(X_train, y_train)

    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svr = svm.SVC()
    clf = grid_search.GridSearchCV(svr, parameters)
    clf.fit(iris.data, iris.target)
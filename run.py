from feature_extraction import *
from train import *


read_images("train_images")
#scaled_X,y = generate_features()
scaled_X,y = load_features()
print(" -------------------------------------------------------------------- ")
svm = Model()
svm.train(scaled_X,y)
#svm.predict()

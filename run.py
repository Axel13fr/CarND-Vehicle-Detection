import matplotlib.pyplot as plt

from feature_extraction import *
from slidding_windows import *
from train import *
from pipeline import *


#read_images("train_images")
#scaled_X,y = generate_features()
scaled_X,y,settings,X_scaler = load_features()
print(" -------------------------------------------------------------------- ")
svc = Model()
#svc.train(scaled_X,y)
svc.load()
#idx = np.random.randint(0,17000)
#print(svm.predict(scaled_X[idx:idx+10]))
#print(y[idx:idx+10])

image = mpimg.imread('test_images/test1.jpg')
box_image = np.copy(image)
draw_image = np.copy(image)
# Scaling back to [0,1] just as when reading PNG files
image = image.astype(np.float32)/255

pipeline = Pipeline(svc,X_scaler,settings,debugView=True)

detections = pipeline.find_cars(image, 400, 650, 40, 1.2)
detections += pipeline.find_cars(image, 400, 700, 30, 1.6)
# Draw results
window_img = draw_boxes(draw_image, detections, color=(0, 0, 255), thick=6)
plt.imshow(window_img)

plt.figure()
plt.imshow(pipeline.filter_detections(draw_image, detections))

plt.figure()
plt.imshow(pipeline.filter_detections(draw_image, detections))

plt.figure()
plt.imshow(pipeline.filter_detections(draw_image, detections))



plt.show()

run_video(svc,X_scaler,settings)

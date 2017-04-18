import matplotlib.pyplot as plt

from feature_extraction import *
from slidding_windows import *
from train import *


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

# Generate windows
windows_1 = slide_window(image, x_start_stop=[None, None], y_start_stop=[350,500],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))
windows_2 = slide_window(image, x_start_stop=[20, None], y_start_stop=[400,550],
                    xy_window=(128, 128), xy_overlap=(0.6, 0.6))
windows_3 = slide_window(image, x_start_stop=[20, None], y_start_stop=[450,None],
                    xy_window=(192, 192), xy_overlap=(0.5, 0.7))


f, (ax1, ax2, ax3) = plt.subplots(3,sharex=True, sharey=True)
ax1.imshow(draw_boxes(box_image,windows_1,color=(0, 255, 0), thick=6))
#box_image = draw_boxes(box_image,windows_1,color=(0, 255, 0), thick=6)
#box_image = draw_boxes(box_image,windows_2,color=(255, 0, 0), thick=6)
ax2.imshow(draw_boxes(box_image,windows_2,color=(255, 0, 0), thick=6))
#box_image = draw_boxes(box_image,windows_3,color=(0, 0, 255), thick=6)
ax3.imshow(draw_boxes(box_image,windows_3,color=(0, 0, 255), thick=6))
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
#plt.imshow(box_image)

# All windows for search
windows = windows_1 + windows_2 + windows_3

# Search inside windows
hot_windows = search_windows(image, windows, svc, X_scaler, settings)
# Draw results
window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
plt.figure()
plt.imshow(window_img)
plt.show()
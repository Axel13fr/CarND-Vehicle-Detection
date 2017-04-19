from slidding_windows import *
from feature_extraction import *
from object_extraction import *
from collections import deque
from line_pipeline import *


class Pipeline():
    def __init__(self,svc,Scaler,settings,debugView=True,line_pipeline=None):
        self.svc = svc
        self.scaler = Scaler
        self.settings = settings
        self.debugView = debugView
        self.line_detection = line_pipeline
        #FIFO with 8 elements max
        self.boxes_history = deque(maxlen=8)

        img_shape = (720, 1280, 3)
        # Generate windows here to go faster in processing:
        windows_1 = slide_window(img_shape, x_start_stop=[None, None], y_start_stop=[350, 500],
                                 xy_window=(64, 64), xy_overlap=(0.5, 0.5))
        windows_2 = slide_window(img_shape, x_start_stop=[20, None], y_start_stop=[400, 550],
                                 xy_window=(128, 128), xy_overlap=(0.6, 0.6))
        windows_3 = slide_window(img_shape, x_start_stop=[20, None], y_start_stop=[450, None],
                                 xy_window=(192, 192), xy_overlap=(0.5, 0.7))
        self.windows = windows = windows_1 + windows_2 + windows_3

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self,img, ystart, ystop, xstart,scale):
        """

        :param img: input image
        :param ystart: search will be done starting from this coordinate
        :param ystop: search will send at this coordinate
        :param xstart: search will be done starting from this (used to recenter the search area)
        :param scale: reduces the whole image while keeping the same window size: used to increase
                      search window size
        :return: a list of all windows where cars were detected
        """
        # Settings shortcuts
        spatial_size = (self.settings.spatial, self.settings.spatial)
        hist_bins = self.settings.histbin

        img_tosearch = img[ystart:ystop,xstart:,:]
        ctrans_tosearch = convert_color(img_tosearch, self.settings.cspace)
        ctrans_tosearch_hsv = convert_color(img_tosearch, 'HSV')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            ctrans_tosearch_hsv = cv2.resize(ctrans_tosearch_hsv, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))


        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.settings.pix_per_cell)
        nyblocks = (ch1.shape[0] // self.settings.pix_per_cell)
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64 # Window size
        nblocks_per_window = (window // self.settings.pix_per_cell)#-1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.settings.orient, self.settings.pix_per_cell,
                                self.settings.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.settings.orient, self.settings.pix_per_cell,
                                self.settings.cell_per_block, feature_vec=False)
        # Windows output
        hot_windows = []

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2))

                xleft = xpos*self.settings.pix_per_cell
                ytop = ypos*self.settings.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                # Color hist expects HSV as input
                subimg_hsv =  cv2.resize(ctrans_tosearch_hsv[ytop:ytop+window, xleft:xleft+window], (64,64))
                hist_features = color_hist(subimg_hsv, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = \
                    self.scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = self.svc.predict(test_features)

                # If found something
                if self.debugView:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    hot_windows.append(((xbox_left + xstart, ytop_draw + ystart),
                                        (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)))
                elif test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    hot_windows.append(((xbox_left + xstart, ytop_draw+ystart),
                                  (xbox_left+win_draw + xstart,ytop_draw+win_draw+ystart)))

        return hot_windows

    def filter_detections(self,img_rgb,boxes):

        # Add boxes to history and remove the oldest if more than FIFO size
        self.boxes_history.append(boxes)

        # Produce heat map using boxes history : Add heat to each box in box list
        heat = np.zeros_like(img_rgb[:, :, 0]).astype(np.float)
        for boxes in self.boxes_history:
            heat = add_heat(heat, boxes)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 6)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img_rgb), labels)

        return draw_img

    def process(self,img_rgb):
        # Apply lane detection if available:
        if self.line_detection:
            draw_image = self.line_detection.process(img_rgb)
        else:
            draw_image = np.copy(img_rgb)

        # Scaling back to [0,1] just as when from video file
        img_rgb = img_rgb.astype(np.float32) / 255

        # Detect cars with 2 different slidding window sizes: 64*1.3 and 64*1.5
        detections = self.find_cars(img_rgb, 400, 650, 40, 1.3)
        detections += self.find_cars(img_rgb, 400, 700, 30, 1.5)

        self.filter_detections(img_rgb,detections)
        # Draw results
        #return draw_boxes(draw_image, detections, color=(0, 0, 255), thick=6)
        return self.filter_detections(draw_image, detections)


def run_video(svc,Scaler,settings):
    from moviepy.editor import VideoFileClip
    output = 'project_output.mp4'
    clip2 = VideoFileClip('project_video.mp4')
    line_pipeline = LinePipeline(debugView=False)
    pipeline = Pipeline(svc,Scaler,settings,debugView=False,line_pipeline=line_pipeline)
    challenge_clip = clip2.fl_image(pipeline.process)
    challenge_clip.write_videofile(output, audio=False)

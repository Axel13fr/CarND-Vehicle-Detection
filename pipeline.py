from slidding_windows import *
from feature_extraction import *


class Pipeline():
    def __init__(self,svc,Scaler,settings,debugView=True):
        self.svc = svc
        self.scaler = Scaler
        self.settings = settings
        self.debugView = debugView

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
        # Settings shortcuts
        spatial_size = (self.settings.spatial, self.settings.spatial)
        hist_bins = self.settings.histbin
        draw_img = np.copy(img)

        img_tosearch = img[ystart:ystop,xstart:,:]
        ctrans_tosearch = convert_color(img_tosearch, self.settings.cspace)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.settings.pix_per_cell)-1
        nyblocks = (ch1.shape[0] // self.settings.pix_per_cell)-1
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
        #hog3 = get_hog_features(ch3, self.settings.orient, self.settings.pix_per_cell,
        #                        self.settings.cell_per_block, feature_vec=False)
        print("nblocks_per_window",nblocks_per_window," hog1 shape",hog1.shape)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                #hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                #hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                hog_features = np.hstack((hog_feat1, hog_feat2))

                xleft = xpos*self.settings.pix_per_cell
                ytop = ypos*self.settings.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                # Color hist expects RGB as input
                subimg_rgb = cv2.cvtColor(subimg,cv2.COLOR_YUV2RGB)
                hist_features = color_hist(subimg_rgb, nbins=hist_bins)
                #hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = \
                    self.scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = self.svc.predict(test_features)

                # If found something
                if test_prediction == 1:
                #if True:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left + xstart, ytop_draw+ystart),
                                  (xbox_left+win_draw + xstart,ytop_draw+win_draw+ystart),
                                  (0,0,1.0),6)

        return draw_img

    def process(self,img_rgb):
        # Scaling back to [0,1] just as when from video file
        img_rgb = img_rgb.astype(np.float32) / 255
        hot_windows = search_windows(img_rgb, self.windows, self.svc, self.scaler, self.settings)
        draw_image = np.copy(img_rgb)*255
        return draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)


def run_video(svc,Scaler,settings):
    from moviepy.editor import VideoFileClip
    output = 'project_output.mp4'
    clip2 = VideoFileClip('project_video.mp4')

    pipeline = Pipeline(svc,Scaler,settings,debugView=True)
    challenge_clip = clip2.fl_image(pipeline.process)
    challenge_clip.write_videofile(output, audio=False)

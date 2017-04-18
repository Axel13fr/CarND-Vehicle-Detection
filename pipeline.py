from slidding_windows import *


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
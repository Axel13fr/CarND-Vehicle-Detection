import pickle
import cv2
import numpy as np

# Camera calibration
class Calibration():
    def __init__(self):
        self.mtx = None
        self.dist = None
    def calibrate(self):
        """
        Opens calibration image and find chessboard corners in image
        :return: calibration matrix & distance
        """

        # prepare object points
        nx = 9
        ny = 6

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = ["camera_cal/calibration2.jpg","camera_cal/calibration3.jpg" ]

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            print("Processing file " + fname)
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                write_name = 'corners_found' + fname.split('/')[-1]
                cv2.imwrite('camera_cal/' + write_name, img)
                cv2.imshow(write_name, img)
                cv2.waitKey(500)
            else:
                print("Error: no chessboard corners found for " + fname)
                assert (False)

        cv2.destroyAllWindows()

        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )

        self.mtx = mtx
        self.dist = dist
        return mtx,dist
    def load(self):
        with open('camera_cal/wide_dist_pickle.p', 'rb') as handle:
            calib = pickle.load(handle)
            self.mtx = calib["mtx"]
            self.dist = calib["dist"]

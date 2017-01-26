import cv2

import os
import pickle
import cv2
import numpy as np


class CameraCalibrator:
    def __init__(self, calibration_images, no_corners_x_dir, no_corners_y_dir):
        """

        :param calibration_images:
        :param no_corners_x_dir:
        :param no_corners_y_dir:
        """
        self.calibration_images = calibration_images
        self.no_corners_x_dir = no_corners_x_dir
        self.no_corners_y_dir = no_corners_y_dir
        self.object_points = []
        self.image_points = []

        self.calibrated_data_path = '../camera_cal/calibrated_data.p'

    def calibrate(self):
        """

        :return:
        """
        object_point = np.zeros((self.no_corners_x_dir * self.no_corners_y_dir, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:self.no_corners_x_dir, 0:self.no_corners_y_dir].T.reshape(-1, 2)

        for idx, file_name in enumerate(self.calibration_images):
            image = cv2.imread(file_name)
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ret, corners = cv2.findChessboardCorners(gray_image,
                                                     (self.no_corners_x_dir, self.no_corners_y_dir),
                                                     None)
            if ret:
                self.object_points.append(object_point)
                self.image_points.append(corners)

        image_size = (image.shape[1], image.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points,
                                                           self.image_points, image_size, None, None)
        calibrated_data = {'mtx': mtx, 'dist': dict}

        with open(self.calibrated_data_path, 'wb') as f:
            pickle.dump(calibrated_data, file=f)

    def undistort(self, image):

        if not os.path.exists(self.calibrated_data_path):
            raise Exception('xxx')

        with open(self.calibrated_data_path, 'rb') as f:
            calibrated_data = pickle.load(file=f)

        image = cv2.imread(image)
        return cv2.undistort(image, calibrated_data['mtx'], calibrated_data['dist'],
                             None, calibrated_data['mtx'])

class PerspectiveTransformer:
    def __init__(self, src_points, dest_points):
        """

        :param src_points:
        :param dest_points:
        """
        self.src_points = src_points
        self.dest_points = dest_points

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dest_points)
        self.M_inverse = cv2.getPerspectiveTransform(self.dest_points, self.src_points)

    def transform(self, image):
        """

        :param image:
        :return:
        """
        size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.M, size, flags=cv2.INTER_LINEAR)

    def inverse_transform(self, image):
        """

        :param image:
        :return:
        """
        size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.M_inverse, size, flags=cv2.INTER_LINEAR)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import glob

    #mport CameraCalibrator.CameraCalibrator

    image = cv2.imread('../test_images/test2.jpg')

    corners = np.float32([[190, 720], [589, 457], [698, 457], [1145, 720]])
    new_top_left = np.array([corners[0, 0], 0])
    new_top_right = np.array([corners[3, 0], 0])
    offset = [150, 0]

    img_size = (image.shape[1], image.shape[0])
    src = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst = np.float32([corners[0] + offset, new_top_left + offset, new_top_right - offset, corners[3] - offset])

    images = glob.glob('../camera_cal/calibration*.jpg')

    calibrator = CameraCalibrator(images, 9, 6);
    # calibrator.calibrate()
    undistorted = calibrator.undistort('../test_images/test2.jpg')
    pers = PerspectiveTransformer(src, dst)
    undistorted = pers.transform(undistorted)



    #plt.imshow(plt.imread('../test_images/straight_lines1.jpg'))
    #plt.plot(corners[0][0], corners[0][1], '.')
    #plt.plot(corners[1][0], corners[1][1], '.')
    #plt.plot(corners[2][0], corners[2][1], '.')
    #plt.plot(corners[3][0], corners[3][1], '.')

    cv2.line(undistorted, (dst[0][0], dst[0][1]), (dst[1][0], dst[1][1]), color=[255, 0, 0], thickness=5)
    plt.imshow(undistorted)

    #plt.plot(dst[0], dst[1], 'k-')


    plt.show()





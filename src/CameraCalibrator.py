import cv2
import numpy as np


class CameraCalibrator:
    def __init__(self, calibration_images, no_corners_x_dir, no_corners_y_dir):
        """

        :param calibration_images:
        :param no_corners_x_dir:
        :param no_corners_y_dir:
        """
        self.calib_images = calibration_images
        self.no_coners_x_dir = no_corners_x_dir
        self.no_coners_y_dir = no_corners_y_dir
        self.object_points = []
        self.image_points = []

    def calibrate(self):
        """

        :return:
        """
        object_point = np.zeros((self.no_coners_x_dir * self.no_coners_y_dir, 3), np.float)
        object_point[:, :2] = np.mgrid[0:self.no_coners_x_dir, 0:self.no_coners_y_dir].T.reshape(-1, 2)

        for idx, file_name in enumerate(self.calib_images):
            image = cv2.imread(file_name)
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



import os
import pickle
import numpy as np
import cv2

CAMERA_CALIBRATION_COEFFICIENTS_FILE = '../camera_cal/calibrated_data.p'


class CameraCalibrator:
    def __init__(self, calibration_images, no_corners_x_dir, no_corners_y_dir,
                 use_existing_camera_coefficients=True):
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

        if not use_existing_camera_coefficients:
            self._calibrate()

    def _calibrate(self):
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
        calibrated_data = {'mtx': mtx, 'dist': dist}

        with open(CAMERA_CALIBRATION_COEFFICIENTS_FILE, 'wb') as f:
            pickle.dump(calibrated_data, file=f)

    def undistort(self, image):

        if not os.path.exists(CAMERA_CALIBRATION_COEFFICIENTS_FILE):
            raise Exception('Camera calibration data file does not exist at ' +
                            CAMERA_CALIBRATION_COEFFICIENTS_FILE)

        with open(CAMERA_CALIBRATION_COEFFICIENTS_FILE, 'rb') as f:
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

    def inverse_transform(self, src_image):
        """

        :param src_image:
        :return:
        """
        size = (src_image.shape[1], src_image.shape[0])
        return cv2.warpPerspective(src_image, self.M_inverse, size, flags=cv2.INTER_LINEAR)


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 10)):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    else:
        abs_sobel = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelxy = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.sqrt(grad_x * grad_x)
    abs_sobely = np.sqrt(grad_y * grad_y)

    absgraddir = np.arctan2(np.absolute(abs_sobely), np.absolute(abs_sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def image_binarize(image, gray_thresh = (20, 255), s_thresh = (170, 255), l_thresh = (30, 255)):
    image_copy = np.copy(image)

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

    hls = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = gray_thresh[0]
    thresh_max = gray_thresh[1]
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = s_thresh[0]
    s_thresh_max = s_thresh[1]

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    #color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    l_binary = np.zeros_like(l_channel)
    l_thresh_min = l_thresh[0]
    l_thresh_max = l_thresh[1]
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    # Combine the two binary thresholds
    channels = 255 * np.dstack((l_binary, sxbinary, s_binary)).astype('uint8')
    binary = np.zeros_like(sxbinary)
    binary[((l_binary == 1) & (s_binary == 1) | (sxbinary == 1))] = 1
    binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')
    #return binary, channels

    return binary


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    import glob

    # # mport CameraCalibrator.CameraCalibrator
    #
    # image = cv2.imread('../test_images/test2.jpg')
    #
    # plt.imshow(plt.imread('../test_images/test2.jpg'))
    # plt.show()
    #
    # corners = np.float32([[190, 720], [589, 457], [698, 457], [1145, 720]])
    # new_top_left = np.array([corners[0, 0], 0])
    # new_top_right = np.array([corners[3, 0], 0])
    # offset = [150, 0]
    #
    # img_size = (image.shape[1], image.shape[0])
    # src = np.float32([corners[0], corners[1], corners[2], corners[3]])
    # dst = np.float32([corners[0] + offset, new_top_left + offset, new_top_right - offset, corners[3] - offset])
    #
    # images = glob.glob('../camera_cal/calibration*.jpg')
    #
    # calibrator = CameraCalibrator(images, 9, 6, use_existing_camera_coefficients=False);
    # undistorted = calibrator.undistort('../camera_cal/calibration1.jpg')
    # pers = PerspectiveTransformer(src, dst)
    # undistorted = pers.transform(undistorted)
    #
    # # plt.imshow(plt.imread('../test_images/straight_lines1.jpg'))
    # # plt.plot(corners[0][0], corners[0][1], '.')
    # # plt.plot(corners[1][0], corners[1][1], '.')
    # # plt.plot(corners[2][0], corners[2][1], '.')
    # # plt.plot(corners[3][0], corners[3][1], '.')
    #
    # cv2.line(undistorted, (dst[0][0], dst[0][1]), (dst[1][0], dst[1][1]), color=[255, 0, 0], thickness=5)
    # plt.imshow(undistorted)
    #
    # # plt.plot(dst[0], dst[1], 'k-')
    #
    #
    # plt.show()

    #sample_image_file = '../test_images/test5.jpg'
    #img = mpimg.imread(sample_image_file)
    #img = image_binarize(img, gray_thresh=(50, 100), s_thresh = (170, 255))

    # sample_image_file = '../test_images/test5.jpg'
    # img_two = mpimg.imread(sample_image_file)
    # img_two = image_binarize(img_two, gray_thresh = (20, 100), s_thresh = (120, 255), l_thresh = (40, 255))
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(img, cmap='gray')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(img_two, cmap='gray')
    #
    # plt.show()

    img = plt.imread('../output_images/undistorted_test_images/test1.jpg')
    img_two = image_binarize(img, gray_thresh=(30, 255), s_thresh=(120, 255), l_thresh=(120, 255))
    plt.imshow(img_two)

    plt.show()

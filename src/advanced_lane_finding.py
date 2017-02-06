import glob
import os
import pickle

import cv2
import numpy as np

# we store camera calibration parameters in the following file
CAMERA_CALIBRATION_COEFFICIENTS_FILE = '../camera_cal/calibrated_data.p'


class CameraCalibrator:
    def __init__(self, calibration_images, no_corners_x_dir, no_corners_y_dir,
                 use_existing_camera_coefficients=True):
        """

        This class encapsulates camera calibration process. When creating an instance of
        CameraCalibrator class, if use_existing_camera_coefficients is False,  __calibrate()
        method is called and save camera calibration coefficients.

        :param calibration_images:
            The list of image used for camera calibration

        :param no_corners_x_dir:
            The number of horizontal corners in calibration images

        :param no_corners_y_dir:
            The number of vertical corners in calibration images

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
            Camera calibration coefficients as a python dictionary
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
        """

        :param image:
        :return:
        """

        if not os.path.exists(CAMERA_CALIBRATION_COEFFICIENTS_FILE):
            raise Exception('Camera calibration data file does not exist at ' +
                            CAMERA_CALIBRATION_COEFFICIENTS_FILE)

        with open(CAMERA_CALIBRATION_COEFFICIENTS_FILE, 'rb') as f:
            calibrated_data = pickle.load(file=f)

        # image = cv2.imread(image)
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


def binary_noise_reduction(img, thresh=4):
    """
    Reduces noise of a binary image by applying a filter which counts neighbours with a value
    and only keeping those which are above the threshold.
    :param img: binary image (0 or 1)
    :param thresh: min number of neighbours with value
    :return:
    """
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbours = cv2.filter2D(img, ddepth=-1, kernel=k)
    img[nb_neighbours < thresh] = 0
    return img


def image_binarize(image, gray_thresh=(20, 255), s_thresh=(170, 255), l_thresh=(30, 255)):
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
    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    l_binary = np.zeros_like(l_channel)
    l_thresh_min = l_thresh[0]
    l_thresh_max = l_thresh[1]
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    # Combine the two binary thresholds
    channels = 255 * np.dstack((l_binary, sxbinary, s_binary)).astype('uint8')
    binary = np.zeros_like(sxbinary)
    binary[((l_binary == 1) & (s_binary == 1) | (sxbinary == 1))] = 1
    binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')
    # return binary, channels

    return binary_noise_reduction(binary)


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):

        self.detected = False

        self.fit_left_x = None
        self.fit_right_x = None

        self.MAX_BUFFER_SIZE = 12

        self.buffer_index = 0
        self.iter_counter = 0

        self.buffer_left = np.zeros((self.MAX_BUFFER_SIZE, 720))
        self.buffer_right = np.zeros((self.MAX_BUFFER_SIZE, 720))

        self.perspective = self._build_perspective_transformer()
        self.calibrator = self._build_camera_calibrator()

    @staticmethod
    def _build_perspective_transformer():
        corners = np.float32([[253, 697], [585, 456], [700, 456], [1061, 690]])
        new_top_left = np.array([corners[0, 0], 0])
        new_top_right = np.array([corners[3, 0], 0])
        offset = [50, 0]

        src = np.float32([corners[0], corners[1], corners[2], corners[3]])
        dst = np.float32([corners[0] + offset, new_top_left + offset, new_top_right - offset, corners[3] - offset])

        perspective = PerspectiveTransformer(src, dst)
        return perspective

    @staticmethod
    def _build_camera_calibrator():
        calibration_images = glob.glob('../camera_cal/calibration*.jpg')
        calibrator = CameraCalibrator(calibration_images,
                                      9, 6, use_existing_camera_coefficients=True)
        return calibrator

    def naive_lane_extractor(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :, 0], axis=0)

        # get midpoint of the histogram
        midpoint = np.int(histogram.shape[0] / 2)

        # get left and right halves of the histogram
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # based on number of events, we calculate hight of a window
        nwindows = 9
        window_height = np.int(binary_warped.shape[0] / nwindows)

        # Extracts x and y coordinates of non-zero pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Set current x coordinated for left and right
        leftx_current = leftx_base
        rightx_current = rightx_base

        margin = 75
        minpix = 35

        # save pixel ids in these two lists
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        fity = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        fit_leftx = self.left_fit[0] * fity ** 2 + self.left_fit[1] * fity + self.left_fit[2]
        fit_rightx = self.right_fit[0] * fity ** 2 + self.right_fit[1] * fity + self.right_fit[2]

        self.detected = True

        return fit_leftx, fit_rightx

    def smart_lane_extractor(self, binary_warped):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 75

        left_lane_inds = (
            (nonzerox > (
                self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] - margin)) & (
                nonzerox < (
                    self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (
                self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] - margin)) & (
                nonzerox < (
                    self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        return left_fitx, right_fitx

    def calculate_road_curvature(self, leftx, rightx):
        ploty = np.linspace(0, 719, num=720)
        y_eval = np.max(ploty)

        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                             (1 + (
                                 2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
                                     1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        return left_curverad, right_curverad

    def fill_lane_lines(self, image, fit_leftx, fit_rightx):
        copy_image = np.zeros_like(image)
        fity = np.linspace(0, copy_image.shape[0] - 1, copy_image.shape[0])

        pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(copy_image, np.int_([pts]), (0, 255, 0))

        return copy_image

    def merge_images(self, binary_img, src_image):
        copy_binary = np.copy(binary_img)
        copy_src_img = np.copy(src_image)

        copy_binary_pers = self.perspective.inverse_transform(copy_binary)
        result = cv2.addWeighted(copy_src_img, 1, copy_binary_pers, 0.3, 0)

        return result

    def process(self, image):
        image = np.copy(image)
        undistorted_image = self.calibrator.undistort(image)
        warped_image = self.perspective.transform(undistorted_image)
        binary_image = image_binarize(warped_image)

        if self.detected:
            fit_leftx, fit_rightx = self.smart_lane_extractor(binary_image)
        else:
            fit_leftx, fit_rightx = self.naive_lane_extractor(binary_image)

        self.buffer_left[self.buffer_index] = fit_leftx
        self.buffer_right[self.buffer_index] = fit_rightx

        self.buffer_index += 1
        self.buffer_index %= self.MAX_BUFFER_SIZE

        if self.iter_counter < self.MAX_BUFFER_SIZE:
            self.iter_counter += 1
            ave_left = np.sum(self.buffer_left, axis=0) / self.iter_counter
            ave_right = np.sum(self.buffer_right, axis=0) / self.iter_counter
        else:
            ave_left = np.average(self.buffer_left, axis=0)
            ave_right = np.average(self.buffer_right, axis=0)

        left_curvature, right_curvature = self.calculate_road_curvature(ave_left, ave_right)
        curvature_text = 'Left Curvature: {:.2f} m    Right Curvature: {:.2f} m'.format(left_curvature, right_curvature)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, curvature_text, (100, 50), font, 1, (221, 28, 119), 2)

        filled_image = self.fill_lane_lines(binary_image, ave_left, ave_right)

        merged_image = self.merge_images(filled_image, image)

        return merged_image


if __name__ == '__main__':
    from moviepy.editor import VideoFileClip

    line = Line()
    out_dir = '../'
    output = out_dir + 'processed_project_video.mp4'
    clip = VideoFileClip("../project_video.mp4")
    out_clip = clip.fl_image(line.process)
    out_clip.write_videofile(output, audio=False)

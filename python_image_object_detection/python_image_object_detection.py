# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit
import json
import os

class PythonImageObjectDetection:
    enable_good_features_output = True
    enable_harris_corner_detection_output = True
    enable_orbs_output = True
    enable_orbs_only_on_blue = True     # should really output bgr
    enable_surf_output = True

    def __init__(self, image_file, output_images):
        self._image_file_path = image_file
        self._output_images = output_images
        self.data_for_json = dict()
        self.json = None
        self.img = None
        self.grey = None

    # def save_avg_img_intensity_manual(self):
    #     rows = self.img.shape[0]
    #     cols = self.img.shape[0]
    #     pixel_bgr = 0.0
    #     for row in range(rows):
    #         for col in range(cols):
    #             pixel_bgr += float(self.img[row, col])
    #
    #     avg_pixel_bgr = pixel_bgr / float(rows*cols)
    #     self.data_for_json['avg_pixel_bgr'] = avg_pixel_bgr

    def _rename_file(self, inject):
        filename, file_extension = os.path.splitext(self._image_file_path)
        filename = filename + '_' + inject
        return filename + '' + file_extension

    def _save_avg_img_intensity_manual(self):
        average = np.average(self.grey)
        self.data_for_json['avg_pixel_bgr'] = average

    def _save_image_mean(self):
        mean = cv2.mean(self.img)
        self.data_for_json['mean'] = mean

    def _calc_good_features(self):
        corners = cv2.goodFeaturesToTrack(self.grey, 20, 0.5, 10)
        corners = np.int0(corners)

        if PythonImageObjectDetection.enable_good_features_output:
            output_img = self.img.copy()
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(output_img, (x, y), 3, 255, -1)
            cv2.imwrite(self._rename_file('good_features'), output_img)

    def _calc_harris_corner_detection(self):
        corners = cv2.cornerHarris(self.grey, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)

        if PythonImageObjectDetection.enable_harris_corner_detection_output:
            output_img = self.img.copy()
            output_img[corners > 0.01 * corners.max()] = [0, 0, 255]
            cv2.imwrite(self._rename_file('harris_corners'), output_img)

    def _calc_orbs_keypoints(self):
        cv2.ocl.setUseOpenCL(False)
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(self.img, None)

        if PythonImageObjectDetection.enable_orbs_output:
            output_img = self.img.copy()
            output_img = cv2.drawKeypoints(output_img, kp, des, color=(0, 255, 0), flags=0)
            cv2.imwrite(self._rename_file('orbs'), output_img)

    def _calc_orbs_only_blue(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(hsv, hsv, mask=mask)

        # perform orb detection
        cv2.ocl.setUseOpenCL(False)
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(result, None)

        if PythonImageObjectDetection.enable_orbs_only_on_blue:
            cv2.imwrite(self._rename_file('orbs_blue_original'), hsv)
            cv2.imwrite(self._rename_file('orbs_blue_mask'), mask)
            cv2.imwrite(self._rename_file('orbs_blue_result'), result)

            output_img = cv2.drawKeypoints(result, kp, des, color=(0, 255, 0), flags=0)
            cv2.imwrite(self._rename_file('orbs_blue'), output_img)

    def _calc_surf_keypoints_and_descriptors(self):
        # SURF is a licensed algorithm!!
        surf = cv2.xfeatures2d.SURF_create(5000)
        kp, des = surf.detectAndCompute(self.grey, None)

        if PythonImageObjectDetection.enable_surf_output:
            output_img = cv2.drawKeypoints(self.img, kp, des, (255, 0, 0), 4)
            cv2.imwrite(self._rename_file('surf'), output_img)

    def execute(self):
        self.img = cv2.imread(self._image_file_path, cv2.IMREAD_COLOR)
        self.grey = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self._save_avg_img_intensity_manual()
        self._save_image_mean()

        self._calc_good_features()
        self._calc_harris_corner_detection()
        self._calc_orbs_keypoints()
        self._calc_orbs_only_blue()
        self._calc_surf_keypoints_and_descriptors()



'''
- go through the directory (later)
- run the default object feature detectors
- extract data
- or store data points into json
'''

if __name__ == "__main__":
    detector = PythonImageObjectDetection('../Resources/dog.jpg', True)
    detector.execute()
    print(detector.json)














if False:
    calc_harris_corner_detection()

if False:
    calc_good_features()








#
# for root, dirs, filenames in os.walk(dir):
#     for file in filenames:
#         f = open(dir + file, 'rb')
#
#         with ExtractorEngine(f) as extractor:
#             extractor.run_command_line_exif_tool = True
#             extractor.execute()
#             extractor.bulk_debug_all_print()
#
#
#
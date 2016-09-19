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

    def execute(self):
        self.img = cv2.imread(self._image_file_path, cv2.IMREAD_COLOR)
        self.grey = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self._save_avg_img_intensity_manual()
        self._save_image_mean()

        self._calc_good_features()
        self._calc_harris_corner_detection()



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
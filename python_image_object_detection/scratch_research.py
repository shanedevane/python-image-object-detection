# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit

"""
- parse an image and extract as much object data as possible
- ie. instead of calling out to a API?
- how many edges? (high edges = high detail, city or crowd scene?)
- taken during day or night
- how much pixesl in foreground vs background
- color mean
"""


IMAGE_RESOURCE = '../Resources/dog.jpg'


img = cv2.imread(IMAGE_RESOURCE, cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(IMAGE_RESOURCE, cv2.IMREAD_COLOR)


def avg_img_intensity_manual():
    img = cv2.imread(IMAGE_RESOURCE, cv2.IMREAD_GRAYSCALE)
    rows = img.shape[0]
    cols = img.shape[0]
    pixel_bgr = 0.0
    for row in range(rows):
        for col in range(cols):
            pixel_bgr += float(img[row, col])

    avg_pixel_bgr = pixel_bgr / float(rows*cols)
    return avg_pixel_bgr


def avg_img_intensity_numpy():
    img = cv2.imread(IMAGE_RESOURCE, cv2.IMREAD_GRAYSCALE)
    average = np.average(img)
    return average


def calc_mean():
    img = cv2.imread(IMAGE_RESOURCE, cv2.IMREAD_COLOR)
    mean = cv2.mean(img)
    return mean


def calc_good_features():
    img = cv2.imread(IMAGE_RESOURCE, cv2.IMREAD_COLOR)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(grey, 20, 0.5, 10)
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    plt.imshow(img)
    plt.show()
    # cv2.imshow('corners', img)


def calc_harris_corner_detection():
    img = cv2.imread(IMAGE_RESOURCE, cv2.IMREAD_COLOR)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(grey, 2, 3, 0.04)

    corners = cv2.dilate(corners, None)
    img[corners > 0.01 * corners.max()] = [0, 0, 255]
    cv2.imshow('corners', img)


def calc_surf_keypoints_and_descriptors():
    # SURF is a licensed algorithm!!
    img = cv2.imread(IMAGE_RESOURCE, cv2.IMREAD_GRAYSCALE)
    surf = cv2.xfeatures2d.SURF_create(5000)
    kp, des = surf.detectAndCompute(img, None)
    # print(kp)
    # print(des)
    # print(surf.descriptorSize())
    img2 = cv2.drawKeypoints(img, kp, des, (255, 0, 0), 4)
    cv2.imshow('surf', img2)


def calc_orbs_keypoints():
    img = cv2.imread(IMAGE_RESOURCE, cv2.IMREAD_GRAYSCALE)
    cv2.ocl.setUseOpenCL(False)
    orb = cv2.ORB_create()
    # kp = orb.detect(img, None)
    kp, des = orb.detectAndCompute(img, None)

    img2 = cv2.drawKeypoints(img, kp, des, color=(0, 255, 0), flags=0)
    cv2.imshow('orbs', img2)


# DO FAST NEXT
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html#fast



if True:
    calc_orbs_keypoints()



if False:
    calc_surf_keypoints_and_descriptors()


if False:
    calc_harris_corner_detection()

if False:
    calc_good_features()

if False:
    #(124.68195753094464, 121.6443755826813, 112.67007171242899, 0.0)
    print(calc_mean())




if False:
    # 119.71247862842034: avg "intensity" as the img was open as grayscale
    print(avg_img_intensity_manual())

    # 119.306965429
    print(avg_img_intensity_numpy())


# setup = '''
# from __main__ import avg_img_intensity_manual, avg_img_intensity_numpy
# '''

# print(min(timeit.Timer('avg_img_intensity_manual()', setup=setup).repeat(7, 1000)))


# print(timeit.timeit(avg_img_intensity_manual(), setup="from __main__ import avg_img_intensity_manual"))
# print(timeit.timeit(avg_img_intensity_numpy(), setup="from __main__ import avg_img_intensity_numpy"))


# print(timeit.timeit("avg_img_intensity_manual()", setup="from __main__ import avg_img_intensity_manual"))
#
# print(timeit.timeit("avg_img_intensity_numpy()", setup="from __main__ import avg_img_intensity_numpy"))



# lap = cv2.Laplacian(img, cv2.CV_64F)    # edges

# read to greyscale
# output image onto color


# cv2.imshow('test', img)
# cv2.imshow('test', lap)
cv2.waitKey(0)
cv2.destroyAllWindows()



# plt.imshow(img, cmap='gray')
# plt.show()


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
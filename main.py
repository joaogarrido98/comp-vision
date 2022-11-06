import time

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

victoria = cv2.imread('victoria.jpg')
victoria = cv2.cvtColor(victoria, cv2.COLOR_BGR2GRAY)
victoria_2 = cv2.imread('victoria2.jpg')
victoria_2 = cv2.cvtColor(victoria_2, cv2.COLOR_BGR2GRAY)


# get the convolution - Task 1.1
def convolutions():
    # create a kernel
    kernel = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])

    # create a plot for all the images and show it in gray
    # original picture
    plt.subplot(2, 2, 1),
    plt.imshow(victoria, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    # opencv picture
    cv_filter = cv2.filter2D(victoria, -1, kernel)
    plt.subplot(2, 2, 2), plt.imshow(cv_filter, cmap='gray')
    plt.title('OpenCV'), plt.xticks([]), plt.yticks([])

    # my convoluted picture with zeros padding clipping
    my_filter = convolve2d(victoria, kernel, 'zeros')
    plt.subplot(2, 2, 3), plt.imshow(my_filter, cmap='gray')
    plt.title('MyFilter Zeros'), plt.xticks([]), plt.yticks([])

    # my convoluted picture with mirror padding
    reflect_filter = convolve2d(victoria, kernel, 'reflect')
    plt.subplot(2, 2, 4), plt.imshow(reflect_filter, cmap='gray')
    plt.title('MyFilter Mirror'), plt.xticks([]), plt.yticks([])

    # show the plot
    plt.show()
    return


# execute the convolution on the image with a kernel and a padding type
# padding types accepted - zeros and all the ones from np.pad() method from numpy
def convolve2d(image, kernel, padding_type):
    # get kernel x(width) and y(height)
    kernel_x = kernel.shape[1]
    kernel_y = kernel.shape[0]

    # get image x(width) and y(height)
    image_x = image.shape[1]
    image_y = image.shape[0]

    # get padding according to kernel shape, e.g 3x3 => padding = 1, 5x5 => padding = 2
    padding_x = kernel_x // 2
    padding_y = kernel_y // 2

    # create output array with same size of image
    output = np.zeros(image.shape)

    if padding_type != 'zeros':
        # get padded image with padding given type
        image_padded = np.pad(image, padding_x, padding_type)
    else:
        # get padded image and fill with zeros
        image_padded = np.zeros((image_y + padding_y * 2, image_x + padding_x * 2))

    # add the values of the images to the correct place in the padded image
    image_padded[padding_y:-padding_y, padding_x:-padding_x] = image

    # go through each pixel and do the convolution
    for y in range(image_y):
        for x in range(image_x):
            output[y][x] = np.sum(image_padded[y:y + kernel_y, x:x + kernel_x] * kernel)

    return output


# get the surf and orb keypoints - Task 2.2
def surf_orb():
    # get all the surf and orb images
    victoria_surf = surf(victoria)
    victoria_2_surf = surf(victoria_2)
    victoria_orb = orb(victoria)
    victoria_2_orb = orb(victoria_2)

    # display images on the plot
    plt.subplot(2, 2, 1), plt.imshow(victoria_surf)
    plt.title('Victoria Surf'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(victoria_2_surf)
    plt.title('Victoria 2 Surf'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(victoria_orb)
    plt.title('Victoria Orb'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(victoria_2_orb)
    plt.title('Victoria 2 Orb'), plt.xticks([]), plt.yticks([])
    plt.show()


# get image with surf keypoints
def surf(image):
    # create surf object
    surf_object = cv2.xfeatures2d.SURF_create(600)
    # get keypoints
    keypoints = surf_object.detect(image, None)
    # draw keypoints on the image
    image_surf = cv2.drawKeypoints(image, keypoints, None)
    return image_surf


# get image with orb keypoints
def orb(image):
    # create orb object
    orb_object = cv2.ORB_create(1000)
    # get keypoints
    keypoints = orb_object.detect(image, None)
    # draw keypoints on the image
    image_orb = cv2.drawKeypoints(image, keypoints, None)
    return image_orb


# perform brute form matching - Task 2.3
def brute_force_matching():
    # create brute force matching for sift and surf
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # get time so we can figure out speed performance
    sift_initial = time.time()
    # get the descriptors and keypoints from sift
    victoria_sift = sift_descriptors(victoria)
    victoria_2_sift = sift_descriptors(victoria_2)
    # perform the brute force matching
    sift_matches = bf.match(victoria_sift[1], victoria_2_sift[1])
    sift_matches = sorted(sift_matches, key=lambda x: x.distance)
    sift_final = time.time()

    # get percentage of matches between the pictures
    sift_percentage_matched = len(sift_matches) / (len(victoria_sift[0]) + len(victoria_2_sift[0])) * 100

    # get time so we can figure out speed performance
    surf_initial = time.time()
    # get the descriptors and keypoints from sift
    victoria_surf = surf_descriptors(victoria)
    victoria_2_surf = surf_descriptors(victoria_2)
    # perform the brute force matching
    surf_matches = bf.match(victoria_surf[1], victoria_2_surf[1])
    surf_matches = sorted(surf_matches, key=lambda x: x.distance)
    surf_final = time.time()

    # get percentage of matches between the pictures
    surf_percentage_matched = len(surf_matches) / (len(victoria_surf[0]) + len(victoria_2_surf[0])) * 100

    # get time so we can figure out speed performance
    orb_initial = time.time()
    # get the descriptors and keypoints from orb
    victoria_orb = orb_descriptors(victoria)
    victoria_2_orb = orb_descriptors(victoria_2)
    # create brute force matching for orb
    orb_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # perform the brute force matching
    orb_matches = orb_bf.match(victoria_orb[1], victoria_2_orb[1])
    orb_matches = sorted(orb_matches, key=lambda x: x.distance)
    orb_final = time.time()

    # get percentage of matches between the pictures
    orb_percentage_matched = len(orb_matches) / (len(victoria_orb[0]) + len(victoria_2_orb[0])) * 100

    # get the images with the keypoints matched
    sift_matching = cv2.drawMatches(victoria, victoria_sift[0], victoria_2, victoria_2_sift[0], sift_matches[:10], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    surf_matching = cv2.drawMatches(victoria, victoria_surf[0], victoria_2, victoria_2_surf[0], surf_matches[:10], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    orb_matching = cv2.drawMatches(victoria, victoria_orb[0], victoria_2, victoria_2_orb[0], orb_matches[:10], None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.subplot(2, 2, 1),
    plt.imshow(sift_matching, cmap='gray')
    plt.title('Sift'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2),
    plt.imshow(surf_matching, cmap='gray')
    plt.title('Surf'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 3),
    plt.imshow(orb_matching, cmap='gray')
    plt.title('Orb'), plt.xticks([]), plt.yticks([])

    plt.show()

    # show in console the results for speed and percentage of matches
    print("Sift - speed: " + str(sift_final - sift_initial) + ", total keypoints: " + str(len(victoria_sift[0]) + len(victoria_2_sift[0])) + ", percentage of matches: " + str(sift_percentage_matched))
    print("Surf - speed: " + str(surf_final - surf_initial) + ", total keypoints: " + str(len(victoria_surf[0]) + len(victoria_2_surf[0])) + ", percentage of matches: " + str(surf_percentage_matched))
    print("Orb - speed: " + str(orb_final - orb_initial) + ", total keypoints: " + str(len(victoria_orb[0]) + len(victoria_2_orb[0])) + ", percentage of matches: " + str(orb_percentage_matched))


# get descriptors for sift
def sift_descriptors(image):
    sift_object = cv2.xfeatures2d.SIFT_create()
    kp, des = sift_object.detectAndCompute(image, None)
    return [kp, des]


# get descriptors for surf
def surf_descriptors(image):
    surf_object = cv2.xfeatures2d.SURF_create()
    kp, des = surf_object.detectAndCompute(image, None)
    return [kp, des]


# get descriptors for orb
def orb_descriptors(image):
    orb_object = cv2.ORB_create()
    kp, des = orb_object.detectAndCompute(image, None)
    return [kp, des]

# call whichever method needed for the task
# convolutions()
# surf_orb()
# brute_force_matching()

# Testing is in separate file called testing.py

import cv2 as cv
import numpy as np


def find_pupil(img):
    """Finds pupil on a given image.

    :param img: Grayscale image 

    :return: Tuple with coordinates and radius of enclosing circle.
    """

    P = np.sum(img) / img.size

    pupil_thresh = P / 2
    _, pupil = cv.threshold(img, pupil_thresh, 255, 0)

    # remove eyelashes
    pupil = _morph_open(pupil, r=4, iters=1)

    (x, y), radius = _find_enclosing_circle(pupil)

    # entire iris circle must be inside img
    max_radius = min(x, y, img.shape[0]-y, img.shape[1]-x) - 2
    radius = min(radius, max_radius)

    return (x, y), radius


def find_iris(img, pupil_x, pupil_y, pupil_r):
    """Finds iris on a given image.

    :param img: Grayscale image 
    :param pupil_x: X coordinate of already detected pupil 
    :param pupil_y: Y coordinate of already detected pupil 
    :param pupil_r: Radius of already detected pupil 

    :return: Tuple with coordinates and radius of enclosing circle.
    """
    height, width = img.shape

    # starting point for searching iris outer edge
    baseline_r = round(2 * pupil_r)

    circle_mask = np.zeros_like(img)
    cv.circle(circle_mask, (pupil_x, pupil_y), baseline_r, 1, 1)

    color_brightness_baseline = np.sum(circle_mask * img) / np.sum(circle_mask)
    color_brightness_threshold = color_brightness_baseline * 1.03

    max_r = min(pupil_x, pupil_y, height-pupil_y, width-pupil_x)  # img borders
    for iris_r in range(baseline_r, max_r):
        circle_mask = np.zeros_like(img)
        cv.circle(circle_mask, (pupil_x, pupil_y), iris_r, 1, 1)
        brightness = np.sum(circle_mask * img) / np.sum(circle_mask)

        if brightness > color_brightness_threshold:
            return (pupil_x, pupil_y), iris_r

    return (pupil_x, pupil_y), min(2 * pupil_r, max_r) - 1


def _morph_open(img, r=3, iters=1):
    """Apply morphological 'opening' operation to image.

    :param img: Binary image 
    :param r: Kernel size
    :param iters: Number of times operation will be applied

    :return: Input image after appling all operations.
    """
    image = cv.bitwise_not(img)
    kernel = np.ones((r, r), np.uint8)
    return cv.bitwise_not(cv.morphologyEx(image, kernel=kernel, iterations=iters, op=cv.MORPH_OPEN))


def _find_enclosing_circle(img):
    """Finds enclosing circle around largest object on binary image.

    :param img: Binary image 

    :return: Tuple with coordinates and radius of enclosing circle.
    """
    img = cv.bitwise_not(img)

    contours, _ = cv.findContours(
        img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv.contourArea)

    (x, y), radius = cv.minEnclosingCircle(cnt)
    center = (round(x), round(y))
    radius = round(radius)

    return center, radius

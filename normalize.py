import cv2 as cv
import numpy as np

from detect import find_pupil, find_iris


def normalize_image(img):
    """Normalize eye image to unified rectangular shape containing only iris pixels.

    :param img: Eye image to normalize

    :return: Normalized iris.
    """
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurried_image = _gaussian_blur(img_gray)
    (pupil_x, pupil_y), pupil_r = find_pupil(blurried_image)
    (iris_x, iris_y), iris_r = find_iris(
        blurried_image, pupil_x, pupil_y, pupil_r)

    cropped_iris = img[iris_y-iris_r:iris_y+iris_r,
                       iris_x-iris_r:iris_x+iris_r]
    unrolled_iris = _unroll_iris(cropped_iris, iris_r, pupil_r)

    return unrolled_iris


def _gaussian_blur(img):
    """Apply gaussian blur to image.

    :param img: Image to blur

    :return: Blurred image.
    """
    gauss_matrix = np.array([
        [0.037, 0.039, 0.04, 0.039, 0.037],
        [0.039, 0.042, 0.042, 0.042, 0.039],
        [0.04, 0.042, 0.043, 0.042, 0.04],
        [0.039, 0.042, 0.042, 0.042, 0.039],
        [0.037, 0.039, 0.04, 0.039, 0.037]
    ])

    return cv.filter2D(img, -1, gauss_matrix)


def _unroll_iris(cropped_iris, r_outer, r_inner, M=64, N=512):
    """Transform iris image from annulus to rectangular shape.

    :param img: Cropped BGR iris image (square cropping)

    :return: ndarray(N, M) with straightened iris image.
    """
    inner_radius_factor = r_inner / r_outer

    warped_width = N
    warped_height = round(M / (1 - inner_radius_factor))

    # Unwarp ring
    warped = cv.warpPolar(
        cropped_iris, (warped_height, warped_width), (r_outer, r_outer), r_outer, 0)

    # Rotate 90 degrees
    straightened = cv.rotate(warped, cv.ROTATE_90_COUNTERCLOCKWISE)

    # Crop to ring only
    cropped = straightened[:M, :]

    return cropped

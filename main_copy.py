import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.util import view_as_blocks


from pathlib import Path

from scipy import ndimage as ndi


from skimage.util import view_as_blocks

image = cv.imread('./CASIA-Iris-Interval/001/L/S1001L01.jpg')


def gaussian_blur(img):
    gauss_matrix = np.array([
        [0.037, 0.039, 0.04, 0.039, 0.037],
        [0.039, 0.042, 0.042, 0.042, 0.039],
        [0.04, 0.042, 0.043, 0.042, 0.04],
        [0.039, 0.042, 0.042, 0.042, 0.039],
        [0.037, 0.039, 0.04, 0.039, 0.037]
    ])

    return cv.filter2D(img, -1, gauss_matrix)


def morph_open(img, r=3, iters=1):
    image = cv.bitwise_not(img)
    kernel = np.ones((r, r), np.uint8)
    return cv.bitwise_not(cv.morphologyEx(image, kernel=kernel, iterations=iters, op=cv.MORPH_OPEN))


def find_enclosing_circle(object):
    object = cv.bitwise_not(object)

    contours, _ = cv.findContours(
        object, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cnt = contours[-1]
    cnt = max(contours, key=cv.contourArea)

    (x, y), radius = cv.minEnclosingCircle(cnt)
    center = (round(x), round(y))
    radius = round(radius)

    return center, radius


def find_pupil(img):
    P = np.sum(img) / (img.shape[0] * img.shape[1])

    pupil_thresh = P / 2

    _, pupil = cv.threshold(img, pupil_thresh, 255, 0)

    # remove eyelashes
    pupil = morph_open(pupil, r=4, iters=2)

    (x, y), radius = find_enclosing_circle(pupil)

    return (x, y), radius


def find_iris(img, pupil_x, pupil_y, pupil_r):
    width, height = img.shape
    # starting point for searching iris outer edge
    baseline_r = round(2 * pupil_r)  # TODO 1.1

    circle_mask = np.zeros_like(img)
    cv.circle(circle_mask, (pupil_x, pupil_y), baseline_r, 1, 1)

    color_brightness_baseline = np.sum(circle_mask * img) / np.sum(circle_mask)
    color_brightness_threshold = color_brightness_baseline * 1.05
    arr = []

    for iris_r in range(baseline_r, height // 2):
        circle_mask = np.zeros_like(img)
        cv.circle(circle_mask, (pupil_x, pupil_y), iris_r, 1, 1)
        brightness = np.sum(circle_mask * img) / np.sum(circle_mask)
        arr.append(brightness)

        if brightness > color_brightness_threshold:
            return (pupil_x, pupil_y), iris_r

    # plt.plot(arr)
    # plt.show()
    return (pupil_x, pupil_y), 2 * pupil_r
    # raise Exception('Iris not found')


# def unroll_iris(cropped_iris, r_outer, r_inner):
#     size = cropped_iris.shape[0]

#     r_outer = r_outer
#     inner_radius_factor = r_inner / r_outer

#     # Unwarp ring
#     warped = cv.warpPolar(cropped_iris, (size, int(
#         size * math.pi)), (r_outer, r_outer), r_outer, 0)
#     # warped = cv.warpPolar(cropped_iris, (200, 200),
#     #                       (r_outer, r_outer), r_outer, 0)
#     cv.imshow('warped', warped)
#     # Rotate 90 degrees
#     straightened = cv.rotate(warped, cv.ROTATE_90_COUNTERCLOCKWISE)
#     cv.imshow('straightened', straightened)
#     # Crop to ring only
#     cropped = straightened[: int(
#         straightened.shape[0] * (1 - inner_radius_factor)), :]

#     return cropped

def unroll_iris(cropped_iris, r_outer, r_inner, M=64, N=512):
    size = cropped_iris.shape[0]  # square cropping
    inner_radius_factor = r_inner / r_outer

    warped_width = N
    warped_height = round(M / (1 - inner_radius_factor))

    # Unwarp ring
    warped = cv.warpPolar(
        cropped_iris, (warped_height, warped_width), (r_outer, r_outer), r_outer, 0)
    # warped = cv.warpPolar(cropped_iris, (200, 200),
    #                       (r_outer, r_outer), r_outer, 0)
    # cv.imshow('warped', warped)
    # Rotate 90 degrees
    straightened = cv.rotate(warped, cv.ROTATE_90_COUNTERCLOCKWISE)
    # cv.imshow('straightened', straightened)
    # Crop to ring only
    cropped = straightened[: M, :]

    return cropped


def apply_gabor_filter(normalized_iris):
    ksize = (11, 11)  # Size of the Gabor filter kernel
    sigma = 4.0  # Standard deviation of the Gaussian function used to create the Gabor filter
    theta = np.pi / 2  # Orientation of the Gabor filter
    lmbda = 10.0  # Wavelength of the sinusoidal factor in the Gabor filter
    gamma = 0.5  # Spatial aspect ratio of the Gabor filter
    psi = 0  # Phase offset of the Gabor filter

    kernel = cv.getGaborKernel(ksize, sigma, theta, lmbda, gamma, psi)
    filtered = cv.filter2D(normalized_iris, cv.CV_8UC3, kernel)

    print(normalized_iris.shape)
    print(filtered.shape)
    print(filtered)
    print(filtered.min())
    # cv.imshow('filtered', filtered)

    return filtered


def defined_gabor_kernel(frequency, sigma_x=None, sigma_y=None,
                         n_stds=3, offset=0, theta=0):
    """
    According to the codes of skimage, I directly rewrote the function gabor_kernel.

    Parameters
    ----------
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
    theta : float, optional
        Orientation in radians. If 0, the harmonic is in the x-direction.
    sigma_x, sigma_y : float, optional
        Standard deviation in x- and y-directions. These directions apply to
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that `sigma_x` controls the *vertical* direction.
    n_stds : scalar, optional
        The linear size of the kernel is n_stds (3 by default) standard
        deviations
    offset : float, optional
        Phase offset of harmonic function in radians.
    Returns
    -------
    g : complex array
        Complex filter kernel.
    """

    x0 = np.ceil(max(np.abs(n_stds * sigma_x * np.cos(theta)),
                     np.abs(n_stds * sigma_y * np.sin(theta)), 1))
    y0 = np.ceil(max(np.abs(n_stds * sigma_y * np.cos(theta)),
                     np.abs(n_stds * sigma_x * np.sin(theta)), 1))
    y, x = np.mgrid[-y0:y0 + 1, -x0:x0 + 1]

    g = np.zeros(y.shape, dtype=complex)
    g[:] = np.exp(-0.5 * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= np.cos(2 * np.pi * frequency * ((x ** 2 + y ** 2) ** 0.5))

    return g


def defined_gabor(img, frequency, sigma_x, sigma_y):
    """
    Perform gabor filter on the image using defined kernel.

    Parameters
    ----------
    param img : the input img
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
    sigma_x, sigma_y : float, optional
        Standard deviation in x- and y-directions. These directions apply to
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that `sigma_x` controls the *vertical* direction.
    -------
    Returns   :
    filtered_real, filtered_imag : the filtered image. Because using the evensymmetric
    filter, the filtered_imag is zero.
    """

    g = defined_gabor_kernel(frequency, sigma_x, sigma_y)
    filtered_real = ndi.convolve(img, np.real(g), mode='wrap', cval=0)
    filtered_imag = ndi.convolve(img, np.imag(g), mode='wrap', cval=0)

    return filtered_real, filtered_imag


def normalize_image(img: np.ndarray):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurried_image = gaussian_blur(img_gray)
    (pupil_x, pupil_y), pupil_r = find_pupil(blurried_image)
    (iris_x, iris_y), iris_r = find_iris(
        blurried_image, pupil_x, pupil_y, pupil_r)

    # cv.circle(img, (iris_x, iris_y), iris_r, (0, 255, 0), 1)
    # cv.circle(img, (pupil_x, pupil_y), pupil_r, (0, 255, 0), 1)
    # cv.imshow('img', img)

    cropped_iris = img[iris_y-iris_r:iris_y +
                       iris_r, iris_x-iris_r:iris_x+iris_r]
    unrolled_iris = unroll_iris(cropped_iris, iris_r, pupil_r)

    # cv.imshow('unrolled', unrolled_iris)

    return unrolled_iris


def gabor(rho, phi, w, theta0, r0, alpha, beta):
    """Calculates gabor wavelet.

    :param rho: Radius of the input coordinates
    :param phi: Angle of the input coordinates
    :param w: Gabor wavelet parameter (see the formula)
    :param theta0: Gabor wavelet parameter (see the formula)
    :param r0: Gabor wavelet parameter (see the formula)
    :param alpha: Gabor wavelet parameter (see the formula)
    :param beta: Gabor wavelet parameter (see the formula)
    :return: Gabor wavelet value at (rho, phi)
    """
    return np.exp(-w * 1j * (theta0 - phi)) * np.exp(-(rho - r0) ** 2 / alpha ** 2) * \
        np.exp(-(phi - theta0) ** 2 / beta ** 2)


def gabor_convolve(img, w, alpha, beta, dr, dtheta):
    """Uses gabor wavelets to extract iris features.

    :param img: Image of an iris
    :param w: w parameter of Gabor wavelets
    :param alpha: alpha parameter of Gabor wavelets
    :param beta: beta parameter of Gabor wavelets
    :return: Transformed image of the iris (real and imaginary)
    :rtype: tuple (ndarray, ndarray)
    """
    single_rho = np.tile(np.linspace(0, 1, dr), (dtheta, 1)).T
    rho = np.tile(single_rho, (img.shape[0]//dr, img.shape[1]//dtheta))
    single_x = np.linspace(0, 1, dr)
    x = np.tile(single_x, img.shape[0]//dr)
    single_y = np.linspace(-np.pi, np.pi, dtheta)
    y = np.tile(single_y, img.shape[1]//dtheta)
    xx, yy = np.meshgrid(x, y)
    # rho = np.tile(np.linspace(0, 1, img.shape[0]), (img.shape[1], 1)).T
    # x = np.linspace(0, 1, img.shape[0])
    # y = np.linspace(-np.pi, np.pi, img.shape[1])
    # xx, yy = np.meshgrid(x, y)

    # return rho * img * np.real(gabor(xx, yy, w, 0, 0.5, alpha, beta).T), \
    #     rho * img * np.imag(gabor(xx, yy, w, 0, 0.5, alpha, beta).T)

    gabor_val = gabor(xx, yy, w, 0, 0.5, alpha, beta).T
    return rho * img * np.real(gabor_val), rho * img * np.imag(gabor_val)


def iris_encode(img, dr=16, dtheta=16, alpha=0.4):
    """Encodes the straightened representation of an iris with gabor wavelets.

    :param img: Image of an iris
    :param dr: Width of image patches producing one feature
    :param dtheta: Length of image patches producing one feature
    :param alpha: Gabor wavelets modifier (beta parameter of Gabor wavelets becomes inverse of this number)
    :return: Iris code and its mask
    :rtype: tuple (ndarray, ndarray)
    """
    # mean = np.mean(img)
    # std = img.std()
    norm_iris = (img - img.mean()) / img.std()
    # patches = view_as_blocks(norm_iris, (dr, dtheta))
    # code = np.zeros((patches.shape[0], patches.shape[1]))  # * 3, * 2

    wavelets = gabor_convolve(norm_iris, 16, alpha, 1 / alpha, dr, dtheta)
    wavelets_real = wavelets[0]
    wavelets_imag = wavelets[1]
    wavelets_real = wavelets_real.reshape(
        (wavelets_real.shape[0]//dr, dr, wavelets_real.shape[1]//dtheta, dtheta))
    wavelets_imag = wavelets_imag.reshape(
        (wavelets_imag.shape[0]//dr, dr, wavelets_imag.shape[1]//dtheta, dtheta))

    sum_real = wavelets_real.sum(axis=(1, 3))
    sum_imag = wavelets_imag.sum(axis=(1, 3))

    # do i need to reduce dims? what is the proper way?

    code = np.concatenate((sum_real, sum_imag), axis=0)
    # code = np.concatenate((wavelets_real, wavelets_imag), axis=0)

    # for i, row in enumerate(patches):
    #     for j, p in enumerate(row):
    #         for k, w in enumerate([16]):  # 16, 32
    #             # print(p.shape)
    #             wavelet = gabor_convolve(p, w, alpha, 1 / alpha)
    #             # print(wavelet[0].shape)
    #             code[i + k, j] = np.sum(wavelet[1])  # 3 * i
    #             # code[i + k, 2 * j + 1] = np.sum(wavelet[1])

    code[code >= 0] = 1
    code[code < 0] = 0
    return code


def encode_normalized_image(unrolled_iris):
    unrolled_iris = cv.cvtColor(unrolled_iris, cv.COLOR_BGR2GRAY)

    code = iris_encode(unrolled_iris)

    # code, _ = defined_gabor(
    #     # unrolled_iris, frequency=0.65, sigma_x=4, sigma_y=1.5)  # 0.65, 3, 1.5
    #     unrolled_iris, frequency=0.07, sigma_x=1.5, sigma_y=1.5)

    # _, code = cv.threshold(code, 127, 1, cv.THRESH_BINARY)
    # return code.astype(int)

    # cv.imshow('code', code)
    # cv.imshow('mask', mask)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return code

    # filtered_2, imag_2 = defined_gabor(
    #     unrolled_iris, frequency=0.65, sigma_x=1.5, sigma_y=4)  # 0,65, 4.5, 1.5

    # print(filtered_1.shape)
    # print(filtered_1.min())
    # print(filtered_1.max())
    # print(filtered_1)
    # cv.imshow('filtered1', filtered_1)
    # cv.imshow('filtered2', filtered_2)
    # cv.imshow('filtered3', imag_1)
    # cv.imshow('filtered4', imag_2)
    # filtered_iris = apply_gabor_filter(unrolled_iris)

    # cv.waitKey(0)
    # cv.destroyAllWindows()


def read_data(data_path, eye='L'):
    # returns dict id -> List[Img]
    data = dict()
    data_path = Path(data_path)
    for data_dir in data_path.iterdir():
        person_id = data_dir.name
        images_dir = data_dir / eye
        images = [cv.imread(str(path))
                  for path in images_dir.iterdir() if path.suffix == '.jpg']
        if len(images) >= 2:  # at least one train and one test sample
            data[person_id] = images
    return data


def is_image_viable(img):
    # entire iris is inside image, pupil r > 20
    pass


def compare_codes(code1, code2):
    # hamming_dists = []
    # print(code1.shape)
    # for rotation in range(code1.shape[1]):
    #     rolled_code1 = np.roll(code1, rotation, axis=1)
    #     d = np.sum(np.bitwise_xor(rolled_code1, code2)) / code1.size
    #     hamming_dists.append(d)

    d = np.sum(np.bitwise_xor(code1, code2)) / code1.size
    return d


def compare_codes_2(a, b, rotation=False):
    """Compares two codes and calculates Jaccard index.

    :param a: Code of the first iris
    :param b: Code of the second iris
    :param mask_a: Mask of the first iris
    :param mask_b: Mask of the second iris
    :param rotation: Maximum cyclic rotation of the code. If this argument is greater than zero, the function will
        return minimal distance of all code rotations. If this argument is False, no rotations are calculated.

    :return: Distance between two codes.
    """
    if rotation:
        d = []
        for i in range(-rotation, rotation + 1):
            c = np.roll(b, i, axis=1)
            d.append(np.sum(np.remainder(a + c, 2)) / a.size)
        return np.min(d)
    return np.sum(np.remainder(a + b, 2)) / a.size


iris_dataset_path = './CASIA-Iris-Interval'

data = read_data(iris_dataset_path, eye='L')
train_data = {id: images[0] for id, images in data.items()}
test_data = {id: images[1:] for id, images in data.items()}


def test_one():
    normalized1 = normalize_image(
        cv.imread('./CASIA-Iris-Interval/002/L/S1002L02.jpg'))
    code1 = encode_normalized_image(normalized1)

    normalized2 = normalize_image(
        cv.imread('./CASIA-Iris-Interval/239/L/S1239L01.jpg'))
    code2 = encode_normalized_image(normalized2)

    print(code1.shape)
    print(type(code1))
    print(type(code1[0, 0]))
    print(code1)

    print(compare_codes_2(code1, code2))

    cv.imshow('normalized1', normalized1)
    cv.imshow('normalized2', normalized2)

    cv.waitKey(0)
    cv.destroyAllWindows()


def test_all():
    templates = dict()

    for id, image in train_data.items():
        normalized = normalize_image(image)
        code = encode_normalized_image(normalized)
        templates[id] = code

    correct = wrong = 0
    iters = 20
    for id, images in test_data.items():
        # iters -= 1
        # if iters < 0:
        #     break
        print(id)

        for image in images:
            normalized = normalize_image(image)
            code = encode_normalized_image(normalized)

            dists = []
            for template_id, template_code in templates.items():
                dist = compare_codes_2(code, template_code)
                dists.append((dist, template_id))

            min_dist = min(dists)
            # dists.sort()
            # print(dists[:5])

            if id == min_dist[1]:
                correct += 1
            else:
                wrong += 1

    print('acc:', correct / (correct + wrong))
    print(correct)
    print(wrong)


# test_one()
test_all()

cv.waitKey(0)
cv.destroyAllWindows()

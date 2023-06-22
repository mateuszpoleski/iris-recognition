import numpy as np
import cv2 as cv


def encode_image(img, dr=16, dtheta=16, alpha=0.4):
    """Encodes the straightened representation of an iris with gabor wavelets.

    :param img: Image of normalized iris in BGR
    :param dr: Width of image patches producing one feature, divisor of img.shape[0]
    :param dtheta: Length of image patches producing one feature, divisor of img.shape[1]
    :param alpha: Gabor wavelets modifier (beta parameter of Gabor wavelets becomes inverse of this number)

    :return: Iris code
    """
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    norm_img = (img - img.mean()) / img.std()

    wavelets = _gabor_convolve(norm_img, 16, alpha, 1 / alpha, dr, dtheta)
    wavelets_real = wavelets[0]
    wavelets_imag = wavelets[1]

    # reshape to allow efficient sum of block matrix
    # https://stackoverflow.com/questions/38948603/best-way-to-take-mean-sum-of-block-matrix-in-numpy
    wavelets_real = wavelets_real.reshape(
        (wavelets_real.shape[0]//dr, dr, wavelets_real.shape[1]//dtheta, dtheta))
    wavelets_imag = wavelets_imag.reshape(
        (wavelets_imag.shape[0]//dr, dr, wavelets_imag.shape[1]//dtheta, dtheta))

    sum_real = wavelets_real.sum(axis=(1, 3))
    sum_imag = wavelets_imag.sum(axis=(1, 3))

    code = np.concatenate((sum_real, sum_imag), axis=0)

    code[code >= 0] = 1
    code[code < 0] = 0
    return code


def _gabor(rho, phi, w, theta0, r0, alpha, beta):
    """Calculates gabor wavelet.

    :param rho: Radius of the input coordinates
    :param phi: Angle of the input coordinates
    :param w: Gabor wavelet parameter
    :param theta0: Gabor wavelet parameter
    :param r0: Gabor wavelet parameter
    :param alpha: Gabor wavelet parameter
    :param beta: Gabor wavelet parameter

    :return: Gabor wavelet value at (rho, phi)
    """
    return np.exp(-w * 1j * (theta0 - phi)) * np.exp(-(rho - r0) ** 2 / alpha ** 2) * \
        np.exp(-(phi - theta0) ** 2 / beta ** 2)


def _gabor_convolve(img, w, alpha, beta, dr, dtheta):
    """Uses gabor wavelets to extract iris features.

    Parameters (rho, x, y) are replicated for each of the patches to allow fast matrix operations.

    :param img: Image of an iris
    :param w: w parameter of Gabor wavelets
    :param alpha: alpha parameter of Gabor wavelets
    :param beta: beta parameter of Gabor wavelets
    :param dr: Width of image patches producing one feature
    :param dtheta: Length of image patches producing one feature

    :return: Transformed image of the iris (real and imaginary)
    :rtype: Tuple (ndarray, ndarray)
    """
    rho = np.tile(np.linspace(0, 1, dr), (dtheta, 1)).T
    rho_patches = np.tile(rho, (img.shape[0]//dr, img.shape[1]//dtheta))

    x = np.linspace(0, 1, dr)
    x_patches = np.tile(x, img.shape[0]//dr)
    y = np.linspace(-np.pi, np.pi, dtheta)
    y_patches = np.tile(y, img.shape[1]//dtheta)
    xx, yy = np.meshgrid(x_patches, y_patches)

    gabor_val = _gabor(xx, yy, w, 0, 0.5, alpha, beta).T
    return rho_patches * img * np.real(gabor_val), \
        rho_patches * img * np.imag(gabor_val)

import numpy as np


def compare_codes(a, b, rotation=False):
    """Compares two codes and calculates normalized Hamming distance.

    :param a: Code of the first iris
    :param b: Code of the second iris
    :param rotation: Maximum cyclic rotation of the code. If this argument is greater than zero, the function will
        return minimal distance of all code rotations. If this argument is False, no rotations are calculated.

    :return: Distance between two codes.
    """
    if not rotation:
        return np.count_nonzero(a != b) / a.size

    d = []
    for i in range(-rotation, rotation + 1):
        c = np.roll(b, i, axis=1)
        d.append(np.count_nonzero(a != c) / a.size)
    return np.min(d)

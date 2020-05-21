from skimage.color import rgb2grey
from scipy import fftpack
import numpy as np


def _dict_divide_(dividends, divisors):
    ret = dict()
    for key, dividend in dividends.items():
        ret[key] = dividend / divisors.get(key, 1)
    return ret


def _get_hashable_key_(key):
    if key.squeeze().ndim == 1:
        return tuple(key)
    else:
        return int(key)


def pixel_accuracy(img1, img2):
    if img1.ndim == 3:
        binary_equality = np.all(img1 == img2, axis=2)
    else:
        binary_equality = (img == img2)

    return np.sum(binary_equality) / binary_equality.size


def iou(img1, img2):
    if img1.ndim == 3:
        class_ids = np.vstack((np.unique(img1.reshape(-1, img1.shape[2]), axis=0), np.unique(img2.reshape(-1, img2.shape[2]), axis=0)))
    else:
        class_ids = np.hstack((np.unique(img1.reshape(-1), axis=0), np.unique(img2.reshape(-1), axis=0)))

    intersections = {}
    unions = {}

    for class_id in class_ids:
        if _get_hashable_key_(class_id) in intersections.keys():
            continue

        if img1.ndim == 3:
            img1_pixels = np.all(img1 == class_id, axis=2)
            img2_pixels = np.all(img2 == class_id, axis=2)
        else:
            img1_pixels = (img1 == class_id)
            img2_pixels = (img2 == class_id)

        intersections[_get_hashable_key_(class_id)] = np.sum(np.logical_and(img1_pixels, img2_pixels))
        unions[_get_hashable_key_(class_id)] = np.sum(np.logical_or(img1_pixels, img2_pixels))

    return _dict_divide_(intersections, unions)


def fdbm(image):
    if image.shape[0] == 3:
        grey_img = rgb2grey(np.rollaxis(image, 0, 3))
    else:
        grey_img = rgb2grey(image)

    spectral_rep = fftpack.fft2(grey_img)
    spectral_rep = fftpack.fftshift(spectral_rep)
    magnitude = np.abs(spectral_rep)
    larger_values = magnitude[(magnitude > magnitude.max() / 1000)]
    return len(larger_values) / float(grey_img.shape[0] * grey_img.shape[1])

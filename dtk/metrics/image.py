from skimage.color import rgb2grey
from scipy import fftpack
import numpy as np
import numpy.ma as ma


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


def pixel_accuracy(img1, img2, ignore_value=None):
    if img1.ndim == 3:
        binary_equality = np.all(img1 == img2, axis=2)
    else:
        binary_equality = (img1 == img2)

    matches = np.sum(binary_equality)
    total = binary_equality.size
    if ignore_value is not None:
        ignore_mask = np.logical_or(img1 == ignore_value, img2 == ignore_value)
        matches -= np.sum(np.logical_and(binary_equality, ignore_mask))
        total -= np.sum(ignore_mask)

    return matches / total


def iou(im1, im2, ignore_value=None):
    if ignore_value is not None:
        mask = np.logical_or(im1 == ignore_value, im2 == ignore_value)
        img1 = ma.masked_array(im1, mask=mask)
        img2 = ma.masked_array(im2, mask=mask)
    else:
        img1 = im1
        img2 = im2

    if img1.ndim == 3:
        class_ids = np.unique(np.vstack((np.unique(img1.reshape(-1, img1.shape[2]), axis=0), np.unique(img2.reshape(-1, img2.shape[2]), axis=0))))
    else:
        class_ids = np.unique(np.hstack((np.unique(img1.reshape(-1), axis=0), np.unique(img2.reshape(-1), axis=0))))

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

        intersection = np.sum(np.logical_and(img1_pixels, img2_pixels))
        union = np.sum(np.logical_or(img1_pixels, img2_pixels))

        if union !=0:
            intersections[_get_hashable_key_(class_id)] = intersection
            unions[_get_hashable_key_(class_id)] = union

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

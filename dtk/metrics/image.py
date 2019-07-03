from skimage.color import rgb2grey
from scipy import fftpack
import numpy as np

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

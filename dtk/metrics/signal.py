from scipy.signal import butter, lfilter
import numpy as np


def butter_lowpass(x, cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, x)
    return y


def smoothness(x, cutoff_freq, fs):
    x_power = np.sum(np.square(x), axis=-1)  # Get the power of the original signal
    smoothed_x = butter_lowpass(x, cutoff_freq, fs)
    x_smooth_power = np.sum(np.square(smoothed_x), axis=-1)  # Get the power of the original signal
    return (x_smooth_power / x_power).mean()

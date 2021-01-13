import math
import numpy as np
from scipy.fftpack import rfft
from scipy.stats.mstats import gmean


def calculate_energy(frame):
    energy = np.sum(np.square(abs(frame), dtype=np.int64), dtype=np.int64)
    energy = math.sqrt(energy / len(frame))
    return energy


def calculate_sfm(frame):
    arithmetic_mean = np.mean(frame)
    geometric_mean = gmean(frame)
    if arithmetic_mean == 0 or geometric_mean / arithmetic_mean <= 0:
        sfm = 0
    else:
        sfm = 10 * np.log10(geometric_mean / arithmetic_mean)
    return sfm


def extract_features(signal, fs, frame_duration=0.01):
    energy = []
    dominant_freqs = []
    sfm = []

    frame_length = int(frame_duration * fs)
    signal_length = signal.shape[0]

    # Calculating features (Energy, SFM, and most dominant frequency)
    for i in range(signal_length // frame_length):
        energy.append(calculate_energy(signal[i * frame_length:(i + 1) * frame_length]))

        # Finds the ftt of the frame
        frame_fft = rfft(signal[i * frame_length:(i + 1) * frame_length], 1024)
        power_spectrum = np.abs(frame_fft)

        sfm[i] = calculate_sfm(power_spectrum)

    return energy, dominant_freqs, sfm


class Vad():
    def __init__(self, fs, frame_duration=0.01, energy_prim_thresh=40, f_prim_thresh=185, sf_prim_thresh=5):
        self.fs = fs
        self.frame_duration = frame_duration
        self.energy_prim_thresh = energy_prim_thresh
        self.f_prim_thresh = f_prim_thresh
        self.sf_prim_thresh = sf_prim_thresh

    def __call__(self, signal, f_sampling):
        energy, dominating_freq, sfm = extract_features(signal, self.fs, frame_duration=self.frame_duration)

        # Finding minimum values of the 30 first frames
        min_energy = min(energy[0:29])
        min_sfm = min(sfm[0:29])

        # Setting decision threshold
        thresh_energy = self.energy_prim_thresh * np.log10(min_energy)
        thresh_freq = self.f_prim_thresh
        thresh_sfm = self.sf_prim_thresh

        speech = []
        silence_count = 0

        return speech

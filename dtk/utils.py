from functools import reduce
import operator
import numpy as np
import re
from numpy.linalg import inv
import os, sys
import tempfile


class RegexMapper():
    def __init__(self, regex, group, map=None):
        self.regex = regex
        self.group = group
        self.map = map

    def __getitem__(self, key):
        match = re.search(self.regex, key)
        if self.map is None:
            return match.group(self.group)
        else:
            return self.map[match.group(self.group)]


class RegexDict(dict):
    def __init__(self, regex, group):
        super(RegexDict, self).__init__()
        self.regex = regex
        self.group = group

    def __getitem__(self, key):
        match = re.search(self.regex, key)
        if match is not None:
            projected_key = match.group(self.group)
        else:
            raise KeyError('Key does not conform')

        return super().__getitem__(projected_key)

    def __setitem__(self, key, value):
        match = re.search(self.regex, key)
        if match is not None:
            projected_key = match.group(self.group)
        else:
            raise KeyError('Key does not conform')

        return super().__setitem__(projected_key, value)


def get_temp_path(ext=""):
    file_path = next(tempfile._get_candidate_names()) + ext
    if os.path.exists("/tmp"):  # If tmp exists then prepend to the path
        file_path = "/tmp/" + file_path

    return file_path


def swp_extension(file, ext):
    return os.path.splitext(file)[0] + ext


class suppress_stdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


def args2dict(args):
    return vars(args)


def dict2args(dictionary):
    arg_list = []
    for k in dictionary.keys():
        if type(dictionary[k]) == type(True):
            if dictionary[k] == True:
                arg_list += ["--" + k]
            continue
        elif dictionary[k] is None:
            continue
        elif isinstance(dictionary[k], list):
            arg_list += ["--" + k]
            for subvalue in dictionary[k]:
                arg_list += [subvalue]
            continue

        arg_list += ["--" + k, str(dictionary[k])]
    return arg_list


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def prime_factors(number):
    factor = 2
    factors = []
    while factor * factor <= number:
        if number % factor:
            factor += 1
        else:
            number //= factor
            factors.append(int(factor))
    if number > 1:
        factors.append(int(number))
    return factors


def group_factors(factors, optimal_factor=4):
    groups = []
    group = []
    for f in factors:
        group += [f]
        factor = reduce(operator.mul, group, 1)
        if factor > optimal_factor:
            last_factor = group.pop()
            groups.append(reduce(operator.mul, group, 1))
            group = [last_factor]

    groups.append(reduce(operator.mul, group, 1))
    return sorted(groups)


class LandmarkSmootherRTS():
    def __init__(self, fps=25, process_noise=225, detector_accuracy=25, ignore_value=np.nan):
        self.kf = None
        self.fps = fps
        self.process_noise = process_noise
        self.detector_accuracy = detector_accuracy
        self.ignore_value = ignore_value

    def __call__(self, points):
        p_pred = []
        p_corr = []
        x = []

        untrackable = 0
        for point in points:
            # If the KF is initialized then perform a prediction
            if self.kf is not None:
                self.kf.predict()
                # Keep the pred cov matrices and predictions
                p_pred.append(self.kf.P.copy())

            if ((np.isnan(self.ignore_value) and np.isnan(point).any()) or np.any(point == self.ignore_value)):
                if self.kf is None:
                    untrackable += 1  # These points can't be processed since we can't initialize the KF
                else:
                    p_corr.append(self.kf.P.copy())
                    x.append(self.kf.x.copy())
                continue

            if self.kf is None:
                self.kf = LandmarkTrackerKF(point, fps=self.fps, process_noise=self.process_noise, detector_accuracy=self.detector_accuracy)
                p_corr.append(self.kf.P.copy())
                x.append(self.kf.x.copy())
                continue

            self.kf.update(point)

            # Keep the corrected cov matrices and corrected estimates
            p_corr.append(self.kf.P.copy())
            x.append(self.kf.x.copy())

        smoothed_points = []

        if untrackable == len(points):
            raise ValueError('No valid landmarks to smooth')

        for i in range(len(points) - untrackable - 1, -1, -1):
            if i == len(x) - 1:
                smooth_x = x[i].copy()
                smooth_p = p_corr[i].copy()
            else:
                c = np.matmul(np.matmul(p_corr[i], self.kf.F.T), inv(p_pred[i]))
                smooth_p = p_corr[i] + np.matmul(np.matmul(c, smooth_p - p_pred[i]), c.T)
                for idx in range(self.kf.no_points):
                    smooth_x[idx] = x[i][idx] + np.matmul(c, smooth_x[idx] - np.matmul(self.kf.F, x[i][idx].T))

            smoothed_points.append(np.reshape(smooth_x[:, 0], (self.kf.no_points // self.kf.no_coordinates, self.kf.no_coordinates)).copy())

        smoothed_points += untrackable * [smoothed_points[-1]]
        smoothed_points.reverse()
        return smoothed_points


class LandmarkTrackerKF():
    def __init__(self, starting_points, fps=25, process_noise=225, detector_accuracy=25):
        dt = 1 / fps
        self.no_coordinates = starting_points.shape[-1]
        self.F = np.array([[1, dt], [0, 1]])
        self.Q = process_noise * np.array([[(dt ** 3) / 3, (dt ** 2) / 2], [(dt ** 2) / 2, dt]])
        self.R = detector_accuracy
        self.detector_accuracy = detector_accuracy
        self.x = np.c_[starting_points.flatten(), np.zeros_like(starting_points.flatten())]
        self.no_points = self.x.shape[0]

        # Normally we would need a covariance matrix per point but since all points are assumed to be updated simultaneously with same measurement
        # noise we can assume that the covariance matrix will evolve in the same way for all of them since it doesn't depend on the innovation
        self.P = np.zeros((2, 2))
        self.P[0, 0] = detector_accuracy
        self.P[1, 1] = detector_accuracy
        self.H = np.expand_dims(np.array([1, 0]), 0)

    def reset(self, starting_points):
        self.x = np.c_[starting_points.flatten(), np.zeros_like(starting_points.flatten())]
        self.P[0, 0] = self.detector_accuracy
        self.P[1, 1] = self.detector_accuracy

    def get_current_estimate(self):
        return np.reshape(self.x[:, 0], (self.no_points // self.no_coordinates, self.no_coordinates)).copy()

    def predict(self):
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q

        for i in range(self.no_points):
            self.x[i] = np.matmul(self.F, self.x[i].T)

        return np.reshape(self.x[:, 0], (self.no_points // self.no_coordinates, self.no_coordinates)).copy()

    def update(self, measurement):
        z = measurement.flatten()
        innov_cov = np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R
        kalman_gain = np.matmul(self.P, self.H.T) / innov_cov  # Innovation in this case is scalar so we can just elementwise divide
        self.P = np.matmul(np.eye(2) - np.matmul(kalman_gain, self.H), self.P)

        for i in range(self.no_points):
            innovation = z[i] - np.matmul(self.H, self.x[i])
            self.x[i] = self.x[i] + np.matmul(kalman_gain, innovation)

        return np.reshape(self.x[:, 0], (self.no_points // self.no_coordinates, self.no_coordinates)).copy()

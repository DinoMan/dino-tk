from functools import reduce
import operator
import numpy as np


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


class Point2DTrackerKF():
    def __init__(self, starting_points, fps=25, process_noise=100, detector_accuracy=25):
        dt = 1 / fps
        self.F = np.array([[1, dt], [0, 1]])
        self.Q = process_noise * np.array([[(dt ** 3) / 3, (dt ** 2) / 2], [(dt ** 2) / 2, dt]])
        self.R = detector_accuracy

        self.x = np.c_[starting_points.flatten(), np.zeros_like(starting_points.flatten())]
        self.P = np.zeros((2 * len(starting_points), 2, 2))
        self.P[:, 0, 0] = detector_accuracy
        self.P[:, 1, 1] = detector_accuracy
        self.H = np.expand_dims(np.array([1, 0]), 0)

    def reset(self, starting_points):
        self.x = np.c_[starting_points.flatten(), np.zeros_like(starting_points.flatten())]
        self.P = np.zeros((2 * len(starting_points), 2, 2))

    def get_current_estimate(self):
        no_points = self.P.shape[0]
        return np.reshape(self.x[:, 0], (no_points // 2, 2)).copy()

    def predict(self):
        no_points = self.P.shape[0]
        for i in range(no_points):
            self.x[i] = np.matmul(self.F, self.x[i].T)
            self.P[i] = np.matmul(np.matmul(self.F, self.P[i]), self.F.T) + self.Q

        return np.reshape(self.x[:, 0], (no_points // 2, 2)).copy()

    def update(self, measurement):
        z = measurement.flatten()
        no_points = self.P.shape[0]
        for i in range(no_points):
            innovation = z[i] - np.matmul(self.H, self.x[i])
            innov_cov = np.matmul(np.matmul(self.H, self.P[i]), self.H.T) + self.R
            kalman_gain = np.matmul(self.P[i], self.H.T) / innov_cov  # Innovation in this case is scalar so we can just elementwise divide
            self.x[i] = self.x[i] + np.matmul(kalman_gain, innovation)
            self.P[i] = np.matmul(np.eye(2) - np.matmul(kalman_gain, self.H), self.P[i])
        return np.reshape(self.x[:, 0], (no_points // 2, 2)).copy()

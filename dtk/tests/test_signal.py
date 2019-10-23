import unittest
import dtk.signal as sgn
import numpy as np


class Signal(unittest.TestCase):

    def smoothness(self):
        a = np.rand(100)  # This time series is random noise
        fs = 100
        b = np.sin(2 * np.linspace(0, 2 * np.pi, 100))  # Sine wave with frequency 2Hz
        low_pass_cutoff = 20  # Anything above 20Hz will be considered non-smooth

        a_smoothness = sgn.smoothness(a, low_pass_cutoff, fs)
        b_smoothness = sgn.smoothness(b, low_pass_cutoff, fs)
        self.assertGreater(b_smoothness, a_smoothness, "Random noise should be less smooth than a 2Hz sine wave")




if __name__ == '__main__':
    unittest.main()

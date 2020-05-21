import unittest
import dtk.metrics.image as img
import numpy as np


class Metrics(unittest.TestCase):

    def test_iou(self):
        a = np.zeros((100, 100))
        b = 2 * np.ones((100, 100))
        a[25:75, 25:75] = 2
        ious = img.iou(a, b)
        self.assertEqual(ious[2], 0.25, "The iou for the class should be 0.25")


if __name__ == '__main__':
    unittest.main()

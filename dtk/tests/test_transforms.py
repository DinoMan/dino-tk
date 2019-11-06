import unittest
import dtk.transforms as trf
from PIL import Image
import numpy as np


class Transform(unittest.TestCase):

    def test_random_crop(self):
        img = Image.new('RGB', (10, 10), color='white')
        tf = trf.RandomCrop()
        cropped_img = tf(img)
        self.assertEqual(cropped_img.size, (9, 9), "The image should be cropped by 90%")
        pixels = img.load()
        pixels[5, 5] = (0, 0, 0)
        cropped_img = tf(img)
        self.assertAlmostEqual(np.mean(cropped_img), 251.85, 2, "The middle image should not be cropped")


if __name__ == '__main__':
    unittest.main()

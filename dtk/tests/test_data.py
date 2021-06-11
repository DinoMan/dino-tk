import unittest
import dtk.filesystem as fs
import os

class DataLoading(unittest.TestCase):

    def test_image_loader(self):
        dataset = fs.ImageDataset(os.path.dirname(__file__) + "/data")
        self.assertEqual(len(dataset), 1, "data folder should have 1 image")
        self.assertEqual(dataset[0].size(1), 256, "Image should resize to 256x256")

if __name__ == '__main__':
    unittest.main()

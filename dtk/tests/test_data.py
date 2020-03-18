import unittest
import dtk.filesystem as fs

class DataLoading(unittest.TestCase):

    def test_image_loader(self):
        dataset = fs.ImageDataset("data")
        self.assertEqual(len(dataset), 1, "data folder should have 1 image")
        self.assertEqual(dataset[0].size(1), 256, "Image should resize to 256x256")

if __name__ == '__main__':
    unittest.main()

import unittest
import os
import dtk.utils as utils

class FileManip(unittest.TestCase):

    def test_temp_file_creation(self):
        file_path = utils.get_temp_path()
        self.assertGreater(len(file_path), 0, "Temporary file name not created")
        file_path = utils.get_temp_path(ext=".wtf")
        self.assertEqual(os.path.splitext(file_path)[1], ".wtf", "Temporary file did not have requested extension")

    def test_extension_swap(self):
        file_path = "/dir/to/file.omg"
        new_file_path = utils.swp_extension(file_path, ".lol")
        print(new_file_path)
        self.assertEqual(os.path.splitext(new_file_path)[1], ".lol", "Could not swap the extension")


if __name__ == '__main__':
    unittest.main()

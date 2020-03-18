import unittest
import dtk.filesystem as fs
import tempfile
import os


class Filesystem(unittest.TestCase):

    def test_file_listing(self):
        d = []
        # Create 4 directories
        for i in range(4):
            d.append(tempfile.mkdtemp())

        file_names = ["f1", "f2", "f3"]
        # Create 4 directories Put 3 files in each one but with different extensions
        files = []
        allowed_exts = []
        for i in range(4):
            allowed_exts.append([])
            for f in file_names:
                files.append(tempfile.mkstemp(prefix=(f + "."), dir=d[i]))
                allowed_exts[-1].append(os.path.splitext(files[-1][1])[1])

        # Now add a file that doesn't match
        files.append(tempfile.mkstemp(suffix=".wtf", dir=d[0]))
        matched_files = fs.list_matching_files(d)
        self.assertEqual(len(matched_files["files"]), 3, "3 files should be matched")
        self.assertEqual(set(matched_files["files"]), set(file_names), "the names should also match")

        # Add a new file to each directory with a disallowed extenison
        for i in range(4):
            files.append(tempfile.mkstemp(prefix="f4.", dir=d[i]))

        # Check first without the exts
        matched_files = fs.list_matching_files(d)
        self.assertEqual(len(matched_files["files"]), 4, "4 files should be matched")

        # Now add the allowed extensions
        matched_files = fs.list_matching_files(d, ext=allowed_exts)
        self.assertEqual(len(matched_files["files"]), 3, "3 files should be matched")
        self.assertEqual(set(matched_files["files"]), set(file_names), "the names should also match")


if __name__ == '__main__':
    unittest.main()

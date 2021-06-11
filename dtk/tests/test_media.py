import unittest
import dtk.media as dtm
import torch
import os
import cv2




class MediaIO(unittest.TestCase):

    def test_save_video(self):
        vid = torch.ones((50, 3, 100, 100))
        vid[25:] = -1
        dtm.save_video("test.mp4", vid, fps=50)
        self.assertTrue(os.path.exists("test.mp4"), "Video could not be saved")
        cap = cv2.VideoCapture("test.mp4")
        self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 50, "Video fps does not correspond to what was requested")
        os.remove("test.mp4")

if __name__ == '__main__':
    unittest.main()

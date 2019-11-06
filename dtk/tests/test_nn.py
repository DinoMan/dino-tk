import unittest
import torch
import dtk.nn as dnn

class Padding(unittest.TestCase):

    def test_double_sided_pad(self):
        t = torch.ones([8, 1])
        padded_t = dnn.pad_both_ends(t, 1, 1)
        self.assertEqual(padded_t.size(), (10, 1), "after paddind dimension should be 10, 1")
        self.assertEqual(padded_t[0], 0, "padding should be 0")
        self.assertEqual(padded_t[9], 0, "padding should be 0")

        t = torch.ones([2, 8, 2])
        padded_t = dnn.pad_both_ends(t, 1, 1, dim=1)
        self.assertEqual(padded_t.size(), (2, 10, 2), "after paddind dimension should be 2, 10, 2")
        self.assertEqual(padded_t[1, 9, 1], 0, "padding should be 0")
        self.assertEqual(padded_t[1, 9, 1], 0, "padding should be 0")
        self.assertEqual(padded_t[0, 9, 0], 0, "padding should be 0")
        self.assertEqual(padded_t[0, 9, 0], 0, "padding should be 0")

        self.assertEqual(padded_t[1, 0, 1], 0, "padding should be 0")
        self.assertEqual(padded_t[1, 0, 1], 0, "padding should be 0")
        self.assertEqual(padded_t[0, 0, 0], 0, "padding should be 0")
        self.assertEqual(padded_t[0, 0, 0], 0, "padding should be 0")

    def test_length_padding(self):
        t = torch.ones([8, 1])
        padded_t = dnn.pad(t, 10)
        self.assertEqual(padded_t.size(), (10, 1), "after paddind dimension should be 10, 1")
        self.assertEqual(padded_t[0], 1, "padding should be 0")
        self.assertEqual(padded_t[8], 0, "padding should be 0")
        self.assertEqual(padded_t[9], 0, "padding should be 0")


class Cutting(unittest.TestCase):

    def test_cutting(self):
        t = torch.ones([8, 1])
        cut_seq = dnn.cut_n_stack(t, 2)
        self.assertEqual(cut_seq.size(), (4, 2, 1), "dimension should be (4, 2, 1)")
        self.assertTrue(torch.any(torch.eq(cut_seq, torch.ones((4, 2, 1)))), "Cut should not contain padding")

        t = torch.tensor(range(1, 9))
        cut_seq = dnn.cut_n_stack(t, 3, cutting_stride=2)
        self.assertEqual(cut_seq.size(), (3, 3), "dimension should be (3, 3)")
        self.assertEqual(cut_seq.max(), 7, "maximum should be 7")

        t = torch.tensor(range(1, 9))
        cut_seq = dnn.cut_n_stack(t, 3, cutting_stride=2, pad_samples=1)
        self.assertEqual(cut_seq.size(), (4, 3), "dimension should be (4, 3)")
        self.assertEqual(cut_seq.max(), 8, "maximum should be 8")
        self.assertEqual(cut_seq[3, 2], 0, "last element should be padding")

        t = torch.tensor(range(1, 9))
        cut_seq = dnn.cut_n_stack(t, 4, cutting_stride=3, pad_samples=2)
        self.assertEqual(cut_seq.size(), (3, 4), "dimension should be (3, 4)")
        self.assertEqual(cut_seq.max(), 8, "maximum should be 8")
        self.assertEqual(cut_seq[2, 3], 0, "last element should be padding")
        self.assertEqual(cut_seq[0, 0], 0, "first element should be padding")


class Sampling(unittest.TestCase):

    def test_sampling(self):
        t = torch.zeros([3, 10, 1])
        length = 8
        l = []
        for i in range(3):
            l.append(length)
            for j in range(length):
                t[i, j] = j + 1

            length += 1

        t_samples = dnn.subsample_batch(t, 5)
        self.assertEqual(t_samples.size(), (3, 5, 1), "dimension should be (3, 5, 1)")

        t_samples = dnn.subsample_batch(t, 4)
        self.assertEqual(t_samples.size(), (3, 4, 1), "dimension should be (3, 4, 1)")

        t_samples = dnn.subsample_batch(t, 8, lengths=l)
        self.assertEqual(t_samples.size(), (3, 8, 1), "dimension should be (3, 8, 1)")
        self.assertEqual(t_samples[0, 7], 8, "No elements should correspond to padding")
        self.assertGreaterEqual(t_samples[1, 7], 8, "No elements should correspond to padding")
        self.assertGreaterEqual(t_samples[2, 7], 8, "No elements should correspond to padding")


if __name__ == '__main__':
    unittest.main()

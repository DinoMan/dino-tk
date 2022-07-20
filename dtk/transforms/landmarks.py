import random
import torch


class RandomHorizontalFlipLandmarks(object):
    """
    Flip the landmarks on an image horizontally
    Args:
        canvas_size (tuple): HxW The size of the canvas (this is required so that the flip can take place)
    """

    def __init__(self, canvas_size, p=0.5, deep_copy=False):
        self.centre = canvas_size[1] // 2
        self.deep_copy = deep_copy
        self.p = p

        # Flipped indices
        self.flipped_idxs = (
            list(range(16, -1, -1))  #  chin
            + list(range(26, 16, -1))  # eyebrows
            + list(range(27, 31))  # nose bridge
            + list(range(35, 30, -1))  # nose tip
            + list(range(45, 41, -1))  # left top eyelid
            + list(range(46, 48))  # left bot eyelid
            + list(range(39, 35, -1))  # right top eyelid
            + list(range(40, 42))  # right bot eyelid
            + list(range(54, 47, -1))  # upper lip
            + list(range(59, 54, -1))  # lower lip
            + list(range(64, 59, -1))  # upper inner lip
            + list(range(67, 64, -1))  # lower inner lip
        )

    def __call__(self, lmks):
        if self.deep_copy:
            landmarks = lmks.clone()
        else:
            landmarks = lmks

        if random.random() > self.p:
            return landmarks

        original_size = landmarks.size()

        landmarks = landmarks.view(-1, landmarks.size(-1))
        landmarks[:, 0] = 2 * self.centre - landmarks[:, 0]
        return landmarks.view(original_size)[:, self.flipped_idxs]

    def __repr__(self):
        return self.__class__.__name__ + "(axis={0})".format(self.centre)


class NormaliseLandmarks(object):
    """
    Normalise the landmarks so they are in the [-1,1] range
    Args:
        offset (float): Offset for the landmarks
        scale (float): Scale for the landmarks
    """

    def __init__(self, offset, scale):
        self.offset = offset
        self.scale = scale

    def __call__(self, landmarks):
        if torch.is_tensor(landmarks):
            scaling = torch.Tensor(self.scale)
        else:
            scaling = self.scale

        return 2 * (landmarks / scaling - self.offset) - 1

    def __repr__(self):
        return self.__class__.__name__


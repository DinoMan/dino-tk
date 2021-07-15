class RandomHorizontalFlipLandmarks(object):
    """
    Flip the landmarks on an image horizontally
    Args:
        canvas_size (tuple): HxW The size of the canvas (this is required so that the flip can take place)
    """

    def __init__(self, canvas_size):
        self.centre = canvas_size[1] // 2

    def __call__(self, landmarks):
        original_size = landmarks.size()
        landmarks = landmarks.view(-1, landmarks.size(-1))
        landmarks[:, 0] = 2 * self.centre - landmarks[:, 0]

        return landmarks.view(original_size)

    def __repr__(self):
        return self.__class__.__name__ + "(axis={0})".format(self.centre)

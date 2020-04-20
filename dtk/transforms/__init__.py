import random


class RandomCrop(object):
    """Crops given PIL.Images at a random location to have a region that is proportional
    to the original size
    """

    def __init__(self, proportion=0.9):
        if not isinstance(proportion, tuple):
            self.proportion = (proportion, proportion)

    def __call__(self, source, proportion=None):
        if proportion is None:
            proportion = self.proportion

        try:  # If img is iterable
            img_iterator = iter(source)
        except TypeError:
            img_iterator = iter([source])

        tl_ratio_x = random.uniform(0, 1)
        tl_ratio_y = random.uniform(0, 1)

        target = []
        for img in img_iterator:
            w, h = img.size
            new_w = proportion[0] * w
            new_h = proportion[1] * h

            x_tl = int(tl_ratio_x * (w - new_w))
            y_tl = int(tl_ratio_y * (h - new_h))
            target.append(img.crop((x_tl, y_tl, x_tl + new_w, y_tl + new_h)))

        if len(target) == 1:
            return target[0]
        else:
            return target

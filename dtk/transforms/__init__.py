import random
import numpy as np
import math


class CutMix(object):
    def __init__(self, p=0.5, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.p = p

    def __call__(self, img_batch1, img_batch2):
        mixed_img = img_batch1.clone()

        batch = mixed_img.size(0)
        width = mixed_img.size(2)
        height = mixed_img.size(3)

        for i in range(batch):
            if random.uniform(0, 1) > self.p:
                continue
            ratio = np.random.beta(self.alpha, self.beta)
            tl, br = self.get_bbox(width, height, ratio)
            mixed_img[i, :, tl[0]:br[0], tl[1]:br[1]] = img_batch2[i, :, tl[0]:br[0], tl[1]:br[1]].clone()

        return mixed_img

    def get_bbox(self, w, h, ratio):
        cut_rat = math.sqrt(1.0 - ratio)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        bbxl = int(random.uniform(0, 1) * (w - cut_w))
        bbyt = int(random.uniform(0, 1) * (h - cut_h))

        bbxr = bbxl + cut_w
        bbyb = bbyt + cut_h

        return (bbxl, bbyt), (bbxr, bbyb)


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

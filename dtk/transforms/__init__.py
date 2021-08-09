import random
import numpy as np
import math
import torch
from .video import CenterCropVideo, ToTensorVideo, NormalizeVideo, RandomHorizontalFlipVideo, BinarizeVideo
from .audio import AmplitudeToDB
from .landmarks import RandomHorizontalFlipLandmarks


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


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


def rot_to_ortho6d(rot_matrix):
    return np.concatenate((rot_matrix[:, 0], rot_matrix[:, 1]))


def ortho6d_to_rot(ortho6d):
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]

    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)
    return matrix


def get_transform_matrix(rotation, translation, scale=None):
    batch_size = rotation.size(0)
    num_coordinates = rotation.size(1)

    trans = torch.zeros((batch_size, num_coordinates + 1, num_coordinates + 1), device=rotation.device)
    if scale is None:
        trans[:, :num_coordinates, :num_coordinates] = rotation
    else:
        trans[:, :num_coordinates, :num_coordinates] = scale.unsqueeze(-1).unsqueeze(-1) * rotation
    trans[:, :num_coordinates, num_coordinates] = translation.squeeze()
    trans[:, num_coordinates, num_coordinates] = 1

    return trans


def procrustes(s1, s2):
    if len(s1.size()) < 3:
        s1 = s1.unsqueeze(0)
    if len(s2.size()) < 3:
        s1 = s1.unsqueeze(0)

    coordinates = s1.size(2)

    mu1 = s1.mean(axis=1, keepdims=True)
    mu2 = s2.mean(axis=1, keepdims=True)

    x1 = s1 - mu1
    x2 = s2 - mu2

    var1 = torch.sum(x1 ** 2, dim=1).sum(dim=1)

    cov = x1.transpose(1, 2).bmm(x2)
    u, s, v = torch.svd(cov.float())

    z = torch.eye(u.shape[1], device=s1.device).unsqueeze(0)
    z = z.repeat(u.shape[0], 1, 1)
    z[:, -1, -1] *= torch.sign(torch.det(u.bmm(v.transpose(1, 2)).float()))

    r = v.bmm(z.bmm(u.permute(0, 2, 1)))
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in r.bmm(cov)]) / var1
    t = mu2.view(-1, coordinates, 1) - (scale.unsqueeze(-1).unsqueeze(-1) * (r.bmm(mu1.view(-1, coordinates, 1))))

    return scale, r, t.squeeze()


def transform_landmarks(ref, transformation):
    ret_np = False
    if isinstance(ref, np.ndarray):
        ret_np = True
        ref = torch.from_numpy(ref)
        transformation = torch.from_numpy(transformation)

    ref = ref.view(-1, ref.size(-2), ref.size(-1))
    transformation = transformation.view(-1, transformation.size(-3), transformation.size(-2), transformation.size(-1))

    seq_length = transformation.shape[1]
    no_points = ref.shape[-2]
    coordinates = ref.shape[-1]

    rot_matrix = transformation[:, :, :coordinates, :coordinates]
    out_translation = transformation[:, :, :coordinates, coordinates]

    out_landmarks = torch.bmm(ref[:, None, :, :].repeat(1, seq_length, 1, 1).view(-1, no_points, 3),
                              rot_matrix.view(-1, 3, 3).transpose(1, 2)).contiguous()

    out_landmarks = out_landmarks.view(-1, seq_length, no_points, coordinates) + out_translation[:, :, None, :]

    if ret_np:
        return out_landmarks.squeeze().numpy()
    else:
        return out_landmarks.squeeze()

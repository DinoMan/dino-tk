import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


class DiversityRegularization(nn.Module):
    def __init__(self, l=10, eps=1.0e-5, max=10):
        super(DiversityRegularization, self).__init__()
        self.l = l
        self.eps = eps
        self.max = max

    def forward(self, s1, s2, z1, z2):
        batch_size = s1.size(0)
        sig1 = s1.view(batch_size, -1)
        sig2 = s2.view(batch_size, -1)

        signal_l1 = F.l1_loss(sig1, sig2.detach(), reduction='none').sum(dim=1) / sig1.size(1)
        z_l1 = F.l1_loss(z1.detach(), z2.detach(), reduction='none').sum(dim=1) / z1.size(1)

        norm_err = (signal_l1 / (z_l1 + self.eps)).mean()
        return -self.l * torch.clamp(norm_err, max=self.max)


class Pullaway(nn.Module):
    def __init__(self):
        super(Pullaway, self).__init__()

    def forward(self, z):
        n = z.size(0)
        z_norm = F.normalize(z, p=2, dim=1)
        similarity = torch.matmul(z_norm, z_norm.transpose(1, 0)) ** 2
        return (torch.sum(similarity) - n) / (n * (n - 1))  # The diagonals will add up to n so we subtract them


class GradientPenalty(nn.Module):
    def __init__(self, l=10):
        super(GradientPenalty, self).__init__()
        self.l = l

    def forward(self, real, fake, func, cond=None):
        # The alpha parameter helps interpolate between real and generated data
        alpha = torch.rand([real.size()[0]], device=real.device, requires_grad=True)
        for i in range(real.dim() - 1):
            alpha = alpha.unsqueeze(1)

        alpha = alpha.expand(real.size())
        interpolates = alpha * real + ((1 - alpha) * fake)
        if cond is None:
            func_interpolates = func(interpolates)
        else:  # If we have a condition then feed it to the critic
            func_interpolates = func(interpolates, *cond)

        # get the gradient of the function at the interpolates
        gradients = autograd.grad(outputs=func_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(func_interpolates.size(), device=real.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        # Returns the gradient penalty which is Gr^2 - 1 scaled by the lambda factor
        gradients = gradients.contiguous().view(real.size()[0], -1)
        # calculate the gradient norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        # Return gradient penalty
        return self.l * ((gradients_norm - 1) ** 2).mean()


class KRWassersteinCriterion(nn.Module):
    def __init__(self):
        super(KRWassersteinCriterion, self).__init__()

    def forward(self, real, fake):
        return fake.mean() - real.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=20.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU(inplace=True)

    def forward(self, dist_sq, target):
        losses = 0.5 * (
                target.float() * dist_sq + (1 - target).float() * self.relu(self.margin - dist_sq.sqrt()).pow(2))
        return losses.mean()


class TVLoss(nn.Module):
    def __init__(self, weight=1, source2d=False):
        super(TVLoss, self).__init__()
        self.weight = weight
        self.source2d = source2d

    def forward(self, x):
        batch_size = x.size()[0]
        count = 0
        unnormalised_tv_loss = 0
        if self.source2d:  # If we have image sources then use the 2D version
            count += self._tensor_size(x[:, :, 1:, :])
            count += self._tensor_size(x[:, :, :, 1:])
            unnormalised_tv_loss += torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
            unnormalised_tv_loss += torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        else:
            count += self._tensor_size(x[:, :, :-1])
            unnormalised_tv_loss += torch.sum(torch.abs(x[:, :, 1:] - x[:, :, :-1]))

        return self.weight * unnormalised_tv_loss / (batch_size * count)

    def _tensor_size(self, t):
        if self.source2d:
            return t.size()[1] * t.size()[2] * t.size()[3]
        else:
            return t.size()[1] * t.size()[2]


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, inp1, inp2, lengths=None):
        if lengths is None:
            loss = torch.mean(torch.abs(inp2 - inp1))
        else:
            loss = 0
            nb_tokens = 0
            for idx, seq_len in enumerate(lengths):
                nb_tokens += inp2.squeeze()[idx, :seq_len].numel()
                loss += torch.sum(torch.abs(inp2.squeeze()[idx, :seq_len] - inp1.squeeze()[idx, :seq_len]))

            loss /= nb_tokens

        return loss


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, inp1, inp2, lengths=None):
        if lengths is None:
            loss = torch.mean(torch.pow(inp2 - inp1, 2))
        else:
            loss = 0
            nb_tokens = 0
            for idx, seq_len in enumerate(lengths):
                nb_tokens += inp2.squeeze()[idx, :seq_len].numel()
                loss += torch.sum(torch.pow(inp2.squeeze()[idx, :seq_len] - inp1.squeeze()[idx, :seq_len], 2))

            loss /= nb_tokens

        return loss


class DICE(nn.Module):
    def __init__(self):
        super(DICE, self).__init__()
        self.smooth = 1.

    def forward(self, pred, gt, lengths=None):
        batch_size = pred.size(0)
        if lengths is None:
            dice = (2 * (pred * gt).sum() + self.smooth) / ((pred ** 2).sum() + (gt ** 2).sum() + self.smooth)
            loss = 1 - dice / batch_size
        else:
            loss = 0
            for idx, seq_len in enumerate(lengths):
                dice = (2 * (pred[idx, :seq_len] * gt[idx, :seq_len]).sum() + self.smooth) / (
                        (pred[idx, :seq_len] ** 2).sum() + (gt[idx, :seq_len] ** 2).sum() + self.smooth)
                loss += dice

            loss = 1 - loss / batch_size

        return loss


class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()

    def forward(self, pred, gt, lengths=None):
        batch_size = pred.size(0)
        if lengths is None:
            loss = F.binary_cross_entropy(pred.view(batch_size, -1), gt.view(batch_size, -1))
        else:
            loss = 0
            for idx, seq_len in enumerate(lengths):
                loss += F.binary_cross_entropy(pred[idx, :seq_len].view(batch_size, -1), gt[idx, :seq_len].view(batch_size, -1))

            loss = loss / batch_size

        return loss

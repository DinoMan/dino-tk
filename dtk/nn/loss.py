import torch
import torch.nn as nn
import torch.autograd as autograd


class GradientPenalty(nn.Module):
    def __init__(self, l=10):
        super(GradientPenalty, self).__init__()
        self.l = l

    def forward(self, real, fake, func):
        # The alpha parameter helps interpolate between real and generated data
        alpha = torch.rand([real.size()[0]], device=real.device, requires_grad=True)
        for i in range(real.dim() - 1):
            alpha = alpha.unsqueeze(1)

        alpha = alpha.expand(real.size())
        interpolates = alpha * real + ((1 - alpha) * fake)
        func_interpolates = func(interpolates)

        # get the gradient of the function at the interpolates
        gradients = autograd.grad(outputs=func_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(func_interpolates.size(), device=real.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        # Returns the gradient penalty which is Gr^2 - 1 scaled by the lambda factor
        gradients = gradients.view(real.size()[0], -1)
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

    def forward(self, dist_sq, target):
        losses = 0.5 * (target.float() * dist_sq + (1 - target).float() * F.relu(self.margin - dist_sq.sqrt()).pow(2))
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
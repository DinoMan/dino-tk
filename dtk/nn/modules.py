import torch
import torch.nn as nn

class MedianPool1d(nn.Module):
    def __init__(self, kernel=3, stride=1, padding=0):
        super(MedianPool1d, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.pad = torch.nn.ReflectionPad1d(padding)

    def forward(self, x):
        x = self.pad(x)
        x = x.unfold(2, self.kernel, self.stride)
        return x.median(dim=-1)[0]
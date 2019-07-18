import torch
import torch.nn as nn
from .utils import same_padding


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


class UnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, in_size, kernel_size, stride=1, batch_norm=True):
        super(UnetBlock2D, self).__init__()
        # This ensures that we have same padding no matter if we have even or odd kernels
        padding = same_padding(kernel_size, stride)
        self.dcl1 = nn.ConvTranspose2d(in_channels + skip_channels, in_channels, 3, padding=1, bias=False)
        self.dcl2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding // 2, bias=False)
        if batch_norm:
            self.activation1 = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(True))
            self.activation2 = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(True))
        else:
            self.activation1 = nn.ReLU(True)
            self.activation2 = nn.ReLU(True)

        self.required_channels = out_channels
        self.out_size_required = tuple(x * stride for x in in_size)

    def forward(self, x, s):
        s = s.view(x.size())

        x = torch.cat([x, s], 1)

        x = self.dcl1(x)
        x = self.activation1(x)

        x = self.dcl2(x, output_size=[-1, self.required_channels, self.out_size_required[0], self.out_size_required[1]])
        x = self.activation2(x)
        return x

class UnetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, in_size, kernel_size, stride=1, batch_norm=True):
        super(UnetBlock1D, self).__init__()
        # This ensures that we have same padding no matter if we have even or odd kernels
        padding = same_padding(kernel_size, stride)
        self.dcl1 = nn.ConvTranspose2d(in_channels + skip_channels, in_channels, 3, padding=1, bias=False)
        self.dcl2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding // 2, bias=False)
        if batch_norm:
            self.activation1 = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(True))
            self.activation2 = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(True))
        else:
            self.activation1 = nn.ReLU(True)
            self.activation2 = nn.ReLU(True)

        self.required_channels = out_channels
        self.out_size_required = tuple(x * stride for x in in_size)

    def forward(self, x, s):
        s = s.view(x.size())

        x = torch.cat([x, s], 1)

        x = self.dcl1(x)
        x = self.activation1(x)

        x = self.dcl2(x, output_size=[-1, self.required_channels, self.out_size_required[0], self.out_size_required[1]])
        x = self.activation2(x)
        return x

class Deconv2D(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size, stride=1, batch_norm=True):
        super(Deconv2D, self).__init__()
        # This ensures that we have same padding no matter if we have even or odd kernels
        padding = same_padding(kernel_size, stride)
        self.dcl = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding // 2,
                                      bias=False)

        if batch_norm:
            self.activation = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(True))
        else:
            self.activation = nn.ReLU(True)

        self.required_channels = out_channels
        self.out_size_required = tuple(x * stride for x in in_size)

    def forward(self, x):
        x = self.dcl(x,
                     output_size=[-1, self.required_channels, self.out_size_required[0], self.out_size_required[1]])

        return self.activation(x)

import torch
import torch.nn as nn
import torch
from .utils import same_padding
from .activations import Swish
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
import torch.nn.functional as F
from math import exp
from math import sqrt


def gaussian(window_size, std_dev):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * std_dev ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


class NoiseInjection2D(nn.Module):
    def __init__(self, channel):
        super(NoiseInjection2D, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, x, n):
        return x + self.weight * n


class RGB2GRAY(nn.Module):
    def __init__(self, weights=(1, 1, 1)):
        super(RGB2GRAY, self).__init__()
        self.weights = weights

    def forward(self, x):
        return ((self.weights[0] * x[:, 0] + self.weights[1] * x[:, 1] + self.weights[2] * x[:, 2]) / sum(self.weights)).unsqueeze(1)


class Renormalization2D(nn.Module):
    def __init__(self, old_mean, old_std, new_mean, new_std, scale=1):
        super(Renormalization2D, self).__init__()
        self.register_buffer('old_mean', torch.tensor(old_mean, requires_grad=False))
        self.register_buffer('old_std', torch.tensor(old_std, requires_grad=False))
        self.register_buffer('new_mean', torch.tensor(new_mean, requires_grad=False))
        self.register_buffer('new_std', torch.tensor(new_std, requires_grad=False))
        self.scale = scale

    def extra_repr(self):
        return 'old_mean={}, old_std={}, new_mean={}, new_std={}'.format(
            self.old_mean, self.old_std, self.new_mean, self.new_std)

    def forward(self, x):
        if self.old_std.ndim == 0:
            x = self.scale * (x * self.old_std + self.old_mean)
            x = (x - self.new_mean) / self.new_std
        else:
            x = self.scale * (x * self.old_std[None, :, None, None] + self.old_mean[None, :, None, None])
            x = (x - self.new_mean[None, :, None, None]) / self.new_std[None, :, None, None]

        return x


class EqualizedLR:
    # This is similar to spectral norm and is implemented in a similar fashion as spectral norm in pytorch
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):  # Weight normalization using Kaiming normalization
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualizedLR(name)  # Creates an object which will normalize the weights

        weight = getattr(module, name)  # Store the original weights
        del module._parameters[name]  # Delete them
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))  # Re-register them with a different name
        module.register_forward_pre_hook(fn)  # Register the equalisation call function as a pre-forward step

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equalize_lr(module, name='weight'):
    EqualizedLR.apply(module, name)
    return module


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, channels, style_dim, equalized_lr=False):
        super(AdaptiveInstanceNorm, self).__init__()

        self.norm = nn.InstanceNorm2d(channels)
        if equalized_lr:
            self.style = equalize_lr(nn.Linear(style_dim, 2 * channels))
        else:
            self.style = nn.Linear(style_dim, 2 * channels)

        self.style.bias.data[:channels] = 1
        self.style.bias.data[channels:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class Conv2DMod(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, demod=True, stride=1, eps=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel={}, stride={}'.format(
            self.in_channels, self.out_channels, self.kernel, self.stride)

    def forward(self, x, s):
        b, c, h, w = x.size()

        weights = self.weight[None, :, :, :, :] * (s[:, None, :, None, None] + 1)  # Modulate

        if self.demod:  # Demodulate
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_channels, *ws)

        padding = same_padding(self.kernel, stride=self.stride) // 2
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.out_channels, h, w)
        return x


class GaussianBlur1D(nn.Module):
    def __init__(self, kernel_size, channels, std_dev=1):
        super(GaussianBlur1D, self).__init__()
        self.filter = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=channels, bias=False)

        self.filter.weight.data = gaussian(kernel_size, std_dev).unsqueeze(0).expand(channels, 1, kernel_size).contiguous()  # Set the kernel
        self.filter.weight.requires_grad = False  # The kernel weights are gaussian they are not trainable

        self.window = gaussian(kernel_size, std_dev).unsqueeze(0).expand(channels, 1, kernel_size).contiguous()

    def forward(self, x):
        return self.filter(x)


class GaussianBlur2D(nn.Module):
    def __init__(self, kernel_size, channels, std_dev=1):
        super(GaussianBlur2D, self).__init__()

        window_1d = gaussian(kernel_size, std_dev).unsqueeze(1)
        self.filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=channels, bias=False)

        self.filter.weight.data = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0).expand(channels, 1, kernel_size,
                                                                                                       kernel_size).contiguous()
        self.filter.weight.requires_grad = False  # The kernel weights are gaussian they are not trainable

    def forward(self, x):
        return self.filter(x)


class ResizeConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor=2, bias=True, mode='nearest', spectral_norm=False):
        super(ResizeConv2D, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode=mode)
        if kernel_size % 2 == 1:
            padding = kernel_size // 2
        else:
            padding = (kernel_size // 2, (kernel_size // 2) - 1, kernel_size // 2, (kernel_size // 2) - 1)
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias)
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x, output_size=None):
        x = self.up(x)
        x = self.pad(x)
        x = self.conv(x)

        if output_size is not None:
            return x.view(output_size)

        return x


class VideoDownsizer(nn.Module):
    def __init__(self, new_size):
        super(VideoDownsizer, self).__init__()
        self.new_size = new_size
        self.resizer = nn.AdaptiveAvgPool2d(new_size)

    def forward(self, x):
        old_size = x.size()
        new_size = list(x.size())
        new_size[-2], new_size[-1] = self.new_size[0], self.new_size[1]

        return self.resizer(x.view(-1, old_size[-2], old_size[-1])).view(new_size)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden=[128], norm=None, activation=None, activation_params=[]):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        layer_input = [input_size]
        layer_input.extend(hidden)
        if activation is None:
            activation = nn.ReLU
            activation_params = [True]

        for i in range(0, len(layer_input) - 1):
            if norm is not None:
                self.layers.append(nn.Sequential(nn.Linear(layer_input[i], layer_input[i + 1]),
                                                 norm(layer_input[i + 1]),
                                                 activation(*activation_params)))
            else:
                self.layers.append(nn.Sequential(nn.Linear(layer_input[i], layer_input[i + 1]),
                                                 activation(*activation_params)))

        self.layers.append(nn.Linear(layer_input[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


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
    def __init__(self, in_channels, out_channels, skip_channels, in_size, kernel_size, stride=1, norm=None, spectral_norm=False, bias=False,
                 activation=None, activation_params=[], resize_convs=False, dropout=0):
        super(UnetBlock2D, self).__init__()

        self.dropout1 = None
        self.dropout2 = None

        if dropout > 0:
            self.dropout1 = nn.Dropout(p=dropout)
            self.dropout2 = nn.Dropout(p=dropout)

        if activation is None:
            activation = nn.ReLU

        # This ensures that we have same padding no matter if we have even or odd kernels
        padding = same_padding(kernel_size, stride)
        if spectral_norm:
            if resize_convs:
                self.dcl1 = nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels + skip_channels, in_channels, 3, padding=1, bias=bias))
                self.dcl2 = ResizeConv2D(in_channels, out_channels, kernel_size, scale_factor=stride, bias=bias, spectral_norm=spectral_norm)
            else:
                self.dcl1 = nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels + skip_channels, in_channels, 3, padding=1, bias=bias))
                self.dcl2 = nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                                      stride=stride, padding=padding // 2, bias=bias))
        else:
            if resize_convs:
                self.dcl1 = nn.ConvTranspose2d(in_channels + skip_channels, in_channels, 3, padding=1, bias=bias)
                self.dcl2 = ResizeConv2D(in_channels, out_channels, kernel_size, scale_factor=stride, bias=bias)
            else:
                self.dcl1 = nn.ConvTranspose2d(in_channels + skip_channels, in_channels, 3, padding=1, bias=bias)
                self.dcl2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding // 2, bias=bias)

        if norm is not None:
            self.activation1 = nn.Sequential(norm(in_channels), activation(*activation_params))
            self.activation2 = nn.Sequential(norm(out_channels), activation(*activation_params))
        else:
            self.activation1 = activation(*activation_params)
            self.activation2 = activation(*activation_params)

        self.required_channels = out_channels
        self.out_size_required = tuple(x * stride for x in in_size)

    def forward(self, x, s):
        x = torch.cat([x, s], 1)

        x = self.dcl1(x)
        x = self.activation1(x)
        if self.dropout1 is not None:
            x = self.dropout1(x)

        x = self.dcl2(x, output_size=[-1, self.required_channels, self.out_size_required[0], self.out_size_required[1]])
        x = self.activation2(x)
        if self.dropout2 is not None:
            x = self.dropout2(x)
        return x


class UnetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, in_size, kernel_size, stride=1, norm=None, spectral_norm=False, bias=False,
                 activation=None, activation_params=[]):
        super(UnetBlock1D, self).__init__()
        if activation is None:
            activation = nn.ReLU

        # This ensures that we have same padding no matter if we have even or odd kernels
        padding = same_padding(kernel_size, stride)
        if spectral_norm:
            self.dcl1 = nn.utils.spectral_norm(nn.ConvTranspose1d(in_channels + skip_channels, in_channels, 3, padding=1, bias=bias))
            self.dcl2 = nn.utils.spectral_norm(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride,
                                                                  padding=padding // 2, bias=bias))
        else:
            self.dcl1 = nn.ConvTranspose1d(in_channels + skip_channels, in_channels, 3, padding=1, bias=bias)
            self.dcl2 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride,
                                           padding=padding // 2, bias=bias)
        if norm is not None:
            self.activation1 = nn.Sequential(norm(in_channels), activation(*activation_params))
            self.activation2 = nn.Sequential(norm(out_channels), activation(*activation_params))
        else:
            self.activation1 = activation()
            self.activation2 = activation()

        self.required_channels = out_channels
        self.out_size_required = in_size * stride

    def forward(self, x, s):
        s = s.view(x.size())
        x = torch.cat([x, s], 1)

        x = self.dcl1(x)
        x = self.activation1(x)

        x = self.dcl2(x, output_size=[-1, self.required_channels, self.out_size_required])
        x = self.activation2(x)
        return x


class SelfAttn2D(nn.Module):
    def __init__(self, in_dim, spectral_norm=False, k=8):
        super(SelfAttn2D, self).__init__()
        if spectral_norm:
            self.query_conv = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1))
            self.key_conv = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1))
            self.value_conv = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))
        else:
            self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1)
            self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, retain_attention=False):
        batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batchsize, -1, width * height).permute(0, 2, 1)  # B x C x (W*H)
        proj_key = self.key_conv(x).view(batchsize, -1, width * height)  # B x C x (W*H)
        energy = torch.bmm(proj_query, proj_key)  # matrix multiplication
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, width, height)

        out = self.gamma * out + x

        if retain_attention:
            return out, attention
        else:
            return out


class SelfAttn1D(nn.Module):
    def __init__(self, in_dim, spectral_norm=False, k=8):
        super(SelfAttn1D, self).__init__()
        if spectral_norm:
            self.query_conv = nn.utils.spectral_norm(nn.Conv1d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1))
            self.key_conv = nn.utils.spectral_norm(nn.Conv1d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1))
            self.value_conv = nn.utils.spectral_norm(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))
        else:
            self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1)
            self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1)
            self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, retain_attention=False):
        batchsize, C, length = x.size()
        proj_query = self.query_conv(x).permute(0, 2, 1)  # Transpose
        proj_key = self.key_conv(x)
        energy = torch.bmm(proj_query, proj_key)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batchsize, -1, length)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, length)

        out = self.gamma * out + x
        if retain_attention:
            return out, attention
        else:
            return out


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation=Swish, activation_params=[]):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation1 = activation(*activation_params)
        self.activation2 = activation(*activation_params)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation2(out)

        return out


class ResNet2D(nn.Module):
    def __init__(self, latent, block=ResNetBlock, layers=[2, 2, 2, 2], channels=3, feature_maps=[64, 128, 256, 512], zero_init_residual=False,
                 activation=Swish, act_params=[]):
        super(ResNet2D, self).__init__()
        self.inplanes = 3
        self.channels = channels
        self.feature_maps = feature_maps
        self.latent = latent
        self.resnet_blocks = nn.ModuleList()

        for i, l in enumerate(layers):
            self.resnet_blocks.append(self._make_layer(block, self.feature_maps[i], l, stride=2, activation=activation, activation_params=act_params))
        self.resnet_blocks.append(self._make_layer(block, self.latent, 1, activation=activation, activation_params=act_params))
        self.resnet_blocks.append(nn.AdaptiveAvgPool2d((1, 1)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, activation="swish"):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, activation=activation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, retain_intermediate=[]):
        if retain_intermediate:
            h = {}
            for layer_no, blk in enumerate(self.resnet_blocks):
                x = blk(x)
                layer_size = max(x.size(1), x.size(2))
                if layer_size in retain_intermediate:
                    h[layer_size] = x

            return x.view(-1, self.latent), h
        else:
            for blk in self.resnet_blocks:
                x = blk(x)

            return x.view(-1, self.latent)


class ResNet3D(nn.Module):
    def __init__(self, code_size, block=ResNetBlock, layers=[2, 2, 2], channels=3, feature_maps=[64, 128, 256], window_size=5,
                 zero_init_residual=False, activation=Swish, act_params=[]):
        super(ResNet3D, self).__init__()
        self.inplanes = feature_maps[0]
        self.channels = channels
        self.feature_maps = feature_maps
        self.code_size = code_size
        layers += [2]
        feature_maps += [code_size]

        self.front_end = nn.Sequential(
            nn.Conv3d(channels, self.feature_maps[0], kernel_size=(window_size, 7, 7), stride=(1, 2, 2), padding=(window_size // 2, 3, 3),
                      bias=False),
            nn.BatchNorm3d(self.feature_maps[0]),
            activation(*act_params),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet_blocks = nn.ModuleList()

        for i, l in enumerate(layers):
            if i == 0:
                self.resnet_blocks.append(self._make_layer(block, self.feature_maps[i], l, stride=1))
            else:
                self.resnet_blocks.append(self._make_layer(block, self.feature_maps[i], l, stride=2))
        self.resnet_blocks.append(nn.AdaptiveAvgPool2d((1, 1)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        length = x.size(1)
        x = x.transpose(1, 2)
        x = self.front_end(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, self.feature_maps[0], x.size(3), x.size(4))
        for blk in self.resnet_blocks:
            x = blk(x)

        return x.view(-1, length, self.code_size)


class Deconv2D(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size, stride=1, norm=None, spectral_norm=False, bias=False, activation=None,
                 activation_params=[]):
        super(Deconv2D, self).__init__()
        # This ensures that we have same padding no matter if we have even or odd kernels
        padding = same_padding(kernel_size, stride)
        if spectral_norm:
            self.dcl = nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                                 stride=stride, padding=padding // 2, bias=bias))
        else:
            self.dcl = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding // 2, bias=bias)

        if norm is not None:
            self.bn = norm(out_channels)
        else:
            self.bn = None

        if activation is None:
            self.activation = None
        else:
            self.activation = activation(*activation_params)

        self.required_channels = out_channels
        self.out_size_required = tuple(x * stride for x in in_size)

    def forward(self, x, out_size=None):
        if out_size is None:
            x = self.dcl(x, output_size=[-1, self.required_channels, self.out_size_required[0], self.out_size_required[1]])
        else:
            x = self.dcl(x, output_size=out_size)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Deconv1D(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size, stride=1, norm=None, spectral_norm=False, bias=False, activation=None,
                 activation_params=[]):
        super(Deconv1D, self).__init__()
        # This ensures that we have same padding no matter if we have even or odd kernels
        padding = same_padding(kernel_size, stride)
        if spectral_norm:
            self.dcl = nn.utils.spectral_norm(nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                                                 stride=stride, padding=(padding - padding // 2), bias=bias))
        else:
            self.dcl = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=(padding - padding // 2), bias=bias)

        if norm is not None:
            self.bn = norm(out_channels)
        else:
            self.bn = None

        if activation is None:
            self.activation = None
        else:
            self.activation = activation(*activation_params)

        self.out_size_required = in_size * stride

    def forward(self, x, out_size=None):
        if out_size is None:
            x = self.dcl(x, output_size=[self.out_size_required])
        else:
            x = self.dcl(x, output_size=out_size)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, norm=None, spectral_norm=False, bias=False, activation=None,
                 activation_params=[]):
        super(Conv2D, self).__init__()
        # This ensures that we have same padding no matter if we have even or odd kernels
        padding = same_padding(kernel_size, stride)
        if spectral_norm:
            self.cl = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size,
                                                       stride=stride, padding=(padding - padding // 2), bias=bias))
        else:
            self.cl = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(padding - padding // 2), bias=bias)

        if norm is not None:
            self.bn = norm(out_channels)
        else:
            self.bn = None

        if activation is None:
            self.activation = None
        else:
            self.activation = activation(*activation_params)

    def forward(self, x):
        x = self.cl(x)
        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, norm=None, spectral_norm=False, bias=False, activation=None,
                 activation_params=[]):
        super(Conv1D, self).__init__()
        # This ensures that we have same padding no matter if we have even or odd kernels
        padding = same_padding(kernel_size, stride)
        if spectral_norm:
            self.cl = nn.utils.spectral_norm(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=(padding - padding // 2), bias=bias))
        else:
            self.cl = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=(padding - padding // 2), bias=bias)

        if norm is not None:
            self.bn = norm(out_channels)
        else:
            self.bn = None

        if activation is None:
            self.activation = None
        else:
            self.activation = activation(*activation_params)

    def forward(self, x):
        x = self.cl(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x

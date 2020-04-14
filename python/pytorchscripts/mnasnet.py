import torch
from torch import nn
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

class SqueezeExcitation(nn.Module):

    def __init__(self, num_features, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, reduced_dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, num_features, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class SpatialAttention(nn.Module):
    def __init__(self, in_channel):
        super(SpatialAttention, self).__init__()
        # shape of weighting  [out_channels, in_channels/groups, kH, kW]

        squeeze_filt = torch.ones((1, in_channel, 1, 1)) / in_channel
        self.register_buffer('squeeze_filt', squeeze_filt)

        #expand_filt = torch.ones((in_channel, 1, 1, 1 ))
        self.sa = nn.Sequential(
            nn.Conv2d(1, in_channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = F.conv2d(x, self.squeeze_filt, stride=1)
        return self.sa(out) * x

class MBConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, expand_ratio, kernel_size=3, reduction_ratio=1, no_skip=False, sa=True):
        super(MBConvBlock, self).__init__()
        self.use_residual = in_planes == out_planes and stride == 1 and not no_skip
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = int(in_planes * expand_ratio)
        layers = []

        # pw
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, kernel_size=1)]

        # dw
        layers += [ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim)]

        # se
        if reduction_ratio != 1:
            reduced_dim = max(1, int(in_planes / reduction_ratio))
            if sa:
                layers += [SpatialAttention(hidden_dim)]
            else:
                layers += [SqueezeExcitation(hidden_dim, reduced_dim)]

        # pw-linear
        layers += [
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MnasNetA1(nn.Module):

    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MnasNetA1, self).__init__()

        # make it nn.Sequential
        self.features = nn.Sequential(
            *self.make_layers(width_mult),
            ConvBNReLU(int(320 * width_mult), 1280, kernel_size=1),
        )

        # building classifier
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

        self._initialize_weights()

    @staticmethod
    def make_layers(width_mult=1.0, sa=True):

        settings = [
            # t, c, n, s, k, r
            [1, 16, 1, 1, 3, 1],  # SepConv_3x3
            [6, 24, 2, 2, 3, 1],  # MBConv6_3x3
            [3, 40, 3, 2, 5, 4],  # MBConv3_5x5, SE
            [6, 80, 4, 2, 3, 1],  # MBConv6_3x3
            [6, 112, 2, 1, 3, 4],  # MBConv6_3x3, SE
            [6, 160, 3, 2, 5, 4],  # MBConv6_5x5, SE
            [6, 320, 1, 1, 3, 1]  # MBConv6_3x3
        ]
        features = [ConvBNReLU(3, int(32 * width_mult), 3, stride=2)]

        in_channels = int(32 * width_mult)
        for i, (t, c, n, s, k, r) in enumerate(settings):
            out_channels = int(c * width_mult)
            no_skip = True if i == 0 else False
            for j in range(n):
                stride = s if j == 0 else 1
                features += [MBConvBlock(in_channels, out_channels, stride, t, k, reduction_ratio=r, no_skip=no_skip, sa=sa)]
                in_channels = out_channels

        return features

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
import torch
import torch.nn as nn


AGG_OPS = {
    'psum' : lambda C, stride, scale, affine, repeats=1: ParamSum(C),
    'cat'  : lambda C, stride, scale, affine, repeats=1: ConcatReduce(C, affine=affine, repeats=repeats),
    'deconv_cat'  : lambda C, stride, scale, affine, repeats=1: DeconvConcatReduce(C, scale=scale, affine=affine, repeats=repeats),
    }


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1, affine=True, repeats=1):
        super(SepConv, self).__init__()
        if C_in != C_out:
            assert repeats == 1, "SepConv with C_in != C_out must have only 1 repeat"
        basic_op = lambda: nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=True))
        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx), basic_op())

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class NormConv(nn.Module):

    def __init__(self, C_in, C_out, ksize):
        super(NormConv, self).__init__()
        self.norm_conv = nn.Sequential(
            nn.Conv2d(C_in, C_out, ksize, stride=1, padding=ksize // 2),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.norm_conv(x)


def resize(x1, x2, largest=True):
    if largest:
        if x1.size()[2:] > x2.size()[2:]:
            x2 = nn.Upsample(size=x1.size()[2:], mode='bilinear')(x2)
        elif x1.size()[2:] < x2.size()[2:]:
            x1 = nn.Upsample(size=x2.size()[2:], mode='bilinear')(x1)
        return x1, x2
    else:
        raise NotImplementedError


class ParamSum(nn.Module):

    def __init__(self, C):
        super(ParamSum, self).__init__()
        self.a = nn.Parameter(torch.ones(C))
        self.b = nn.Parameter(torch.ones(C))

    def forward(self, x, y):
        bsize = x.size(0)
        x, y = resize(x, y)
        return (self.a.expand(bsize, -1)[:, :, None, None] * x +
                self.b.expand(bsize, -1)[:, :, None, None] * y)


class ConcatReduce(nn.Module):

    def __init__(self, C, affine=True, repeats=1):
        super(ConcatReduce, self).__init__()
        self.conv1x1 = nn.Sequential(
                            nn.BatchNorm2d(2 * C, affine=affine),
                            nn.ReLU(inplace=False),
                            nn.Conv2d(2 * C, C, 1, stride=1, groups=C, padding=0, bias=False)
                        )

    def forward(self, x, y):
        x, y = resize(x, y)
        z = torch.cat([x, y], 1)
        return self.conv1x1(z)


class DeconvConcatReduce(nn.Module):

    def __init__(self, C, scale=1, affine=True, repeats=1):
        super(DeconvConcatReduce, self).__init__()
        self.upsample = nn.ConvTranspose2d(C, C, scale, stride=scale, padding=0, dilation=1, bias=False)
        self.conv1x1 = nn.Sequential(
                            nn.BatchNorm2d(2 * C, affine=affine),
                            nn.ReLU(inplace=False),
                            nn.Conv2d(2 * C, C, 1, stride=1, groups=C, padding=0, bias=False)
                        )

    def forward(self, x, y):
        if x.size()[2:] > y.size()[2:]:
            y = self.upsample(y)
        elif x.size()[2:] < y.size()[2:]:
            x = self.upsample(x)
        z = torch.cat([x, y], 1)
        return self.conv1x1(z)


class CBR(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):

    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)

    def forward(self, input):
        output = self.conv(input)
        return output


class CDilated(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False, dilation=d, groups=groups)

    def forward(self, input):
        output = self.conv(input)
        return output


class CDilatedB(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False, dilation=d, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        return self.bn(self.conv(input))


class Shuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

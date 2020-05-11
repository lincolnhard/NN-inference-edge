import torch
import math
import torch.nn.functional as F
from torch import nn


def activation_fn(features, name='prelu', inplace=True):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'selu':
        return nn.SELU(inplace=inplace)
    elif name == 'prelu':
        return nn.PReLU(features)
    else:
        NotImplementedError('Not implemented yet')
        exit()


class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, dilation=1, groups=1, act_name='prelu'):
        super().__init__()
        padding = int((kSize - 1) / 2)*dilation
        self.cbr = nn.Sequential(
            nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups, dilation=dilation),
            nn.BatchNorm2d(nOut),
            activation_fn(features=nOut, name=act_name)
        )

    def forward(self, x):
        return self.cbr(x)


class BR(nn.Module):
    def __init__(self, nOut, act_name='prelu'):
        super().__init__()
        self.br = nn.Sequential(
            nn.BatchNorm2d(nOut),
            activation_fn(nOut, name=act_name)
        )

    def forward(self, x):
        return self.br(x)


class Shuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        # print(x.shape)
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        # print(x.shape)
        x = torch.transpose(x, 1, 2).contiguous()
        # print(x.shape)
        x = x.view(batchsize, -1, height, width)
        # print(x.shape)
        return x


class EfficientPWConv(nn.Module):
    def __init__(self, nin, nout):
        super(EfficientPWConv, self).__init__()
        self.wt_layer = nn.Sequential(
                        nn.AdaptiveAvgPool2d(output_size=1),
                        nn.Conv2d(nin, nout, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                        nn.Sigmoid()
                    )

        self.groups = math.gcd(nin, nout)
        self.expansion_layer = CBR(nin, nout, kSize=3, stride=1, groups=self.groups)
        self.out_size = nout
        self.in_size = nin

    def forward(self, x):
        wts = self.wt_layer(x)
        x = self.expansion_layer(x)
        print(x.shape, wts.shape)
        x = x * wts
        return x

    def __repr__(self):
        s = '{name}(in_channels={in_size}, out_channels={out_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class EfficientPyrPool(nn.Module):

    #def __init__(self, in_planes, proj_planes, out_planes, scales=[2.0, 1.5, 1.0, 0.5, 0.1], last_layer_br=True):
    def __init__(self, in_planes, proj_planes, out_planes, scales=[2.0, 1.0, 0.5, 0.1], last_layer_br=True):
        super(EfficientPyrPool, self).__init__()
        scales.sort(reverse=True)

        self.projection_layer = CBR(in_planes, proj_planes, 1, 1)

        stage_layers = [
            nn.Sequential(*[
                nn.ConvTranspose2d(proj_planes, proj_planes, kernel_size=4, stride=2, padding=1, dilation=1, bias=False, groups=proj_planes),
                nn.Conv2d(proj_planes, proj_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=proj_planes),
                nn.AvgPool2d(kernel_size=4, padding=1, stride=2)
            ]),
            nn.Sequential(*[nn.Conv2d(proj_planes, proj_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=proj_planes)]),
            nn.Sequential(*[
                nn.AvgPool2d(kernel_size=4, padding=1, stride=2),
                nn.Conv2d(proj_planes, proj_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=proj_planes),
                nn.ConvTranspose2d(proj_planes, proj_planes, kernel_size=4, stride=2, padding=1, dilation=1, bias=False, groups=proj_planes)
            ]),
            nn.Sequential(*[
                nn.AvgPool2d(kernel_size=12, padding=1, stride=10),
                nn.Conv2d(proj_planes, proj_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=proj_planes),
                nn.ConvTranspose2d(proj_planes, proj_planes, kernel_size=12, stride=10, padding=1, dilation=1, bias=False, groups=proj_planes)
            ])]
        #self.stages = nn.Sequential(*stage_layers)
        self.stages = nn.ModuleList(stage_layers)

        #self.stages = nn.ModuleList()
        #for _ in enumerate(scales):
        #    self.stages.append(nn.Conv2d(proj_planes, proj_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=proj_planes))

        self.merge_layer = nn.Sequential(
            # perform one big batch normalization instead of p small ones
            BR(proj_planes * len(scales)),
            Shuffle(groups=len(scales)),
            CBR(proj_planes * len(scales), proj_planes, 3, 1, groups=proj_planes),
            nn.Conv2d(proj_planes, out_planes, kernel_size=1, stride=1, bias=not last_layer_br),
        )
        if last_layer_br:
            self.br = BR(out_planes)
        self.last_layer_br = last_layer_br
        self.scales = scales

    def forward(self, x):
        hs = []
        x = self.projection_layer(x)
        for stage in self.stages:
            h = stage(x)
            hs.append(h)
        '''
        height, width = x.size()[2:]
        for i, stage in enumerate(self.stages):
            h_s = int(math.ceil(height * self.scales[i]))
            w_s = int(math.ceil(width * self.scales[i]))
            h_s = h_s if h_s > 5 else 5
            w_s = w_s if w_s > 5 else 5
            if self.scales[i] < 1.0:
                h = F.adaptive_avg_pool2d(x, output_size=(h_s, w_s))
                h = stage(h)
                h = F.interpolate(h, (height, width), mode='bilinear', align_corners=True)
            elif self.scales[i] > 1.0:
                h = F.interpolate(x, (h_s, w_s), mode='bilinear', align_corners=True)
                h = stage(h)
                h = F.adaptive_avg_pool2d(h, output_size=(height, width))
            else:
                h = stage(x)
            hs.append(h)
        '''

        out = torch.cat(hs, dim=1)
        out = self.merge_layer(out)
        if self.last_layer_br:
            return self.br(out)
        return out

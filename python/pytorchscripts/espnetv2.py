import torch
from torch import nn
import torch.nn.functional as F
from .layer_factory import CBR, BR, C, CB, CDilated, CDilatedB


ALPHA_MAP = {
    0.5: [16, 32, 64, 128, 256, 1024],
    1.0: [32, 64, 128, 256, 512, 1024],
    1.25: [32, 80, 160, 320, 640, 1024],
    1.5: [32, 96, 192, 384, 768, 1024],
    2.0: [32, 128, 256, 512, 1024, 1280]
}


class DownSampler(nn.Module):

    def __init__(self, nin, nout, k=4, r_lim=9, reinf=True):
        super().__init__()
        config_inp_reinf = 3
        nout_new = nout - nin
        self.eesp = EESP(nin, nout_new, stride=2, k=k, r_lim=r_lim, down_method='avg')
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        if reinf:
            self.inp_reinf = nn.Sequential(
                CBR(config_inp_reinf, config_inp_reinf, 3, 1),
                CB(config_inp_reinf, nout, 1, 1)
            )
        self.act = nn.PReLU(nout)

    def forward(self, downtimes, input, input2=None):
        avg_out = self.avg(input)
        eesp_out = self.eesp(input)
        output = torch.cat([avg_out, eesp_out], 1)
        # print(input.shape, avg_out.shape, eesp_out.shape)
        # print(input.shape, input2.shape, avg_out.size(2))

        if input2 is not None:
            for _ in range(downtimes):
                input2 = self.avg(input2)

            output = output + self.inp_reinf(input2)

        return self.act(output)


class EESP(nn.Module):

    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, down_method='esp'):
        super().__init__()
        self.stride = stride
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)

        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride, groups=n, d=d_rate))
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(nOut)
        self.module_act = nn.PReLU(nOut)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + output[k - 1]
            output.append(out_k)
        # print(output[0].shape, output[1].shape, output[2].shape, output[3].shape, torch.cat(output, 1).shape)
        expanded = self.conv_1x1_exp(self.br_after_cat(torch.cat(output, 1)))
        del output
        if self.stride == 2 and self.downAvg:
            return expanded

        expanded = expanded + input
        return self.module_act(expanded)


class EESPNet(nn.Module):

    def __init__(self, num_classes=1000, channels_in=3, width_mult=1.0,
                 reps_at_each_level = [0, 3, 7, 3], recept_limit = [13, 11, 9, 7, 5], branches = 4):
        super().__init__()

        if not width_mult in ALPHA_MAP.keys():
            print('Model at scale s={} is not suppoerted yet'.format(s))
            exit(-1)

        out_channel_map = ALPHA_MAP[width_mult]
        K = [branches] * len(recept_limit)
        self.input_reinforcement = True

        self.level1 = CBR(channels_in, out_channel_map[0], 3, 2)
        self.level2_0 = DownSampler(out_channel_map[0], out_channel_map[1], k=K[0], r_lim=recept_limit[0], reinf=self.input_reinforcement)
        self.level3_0 = DownSampler(out_channel_map[1], out_channel_map[2], k=K[1], r_lim=recept_limit[1], reinf=self.input_reinforcement)

        self.level3 = nn.ModuleList()
        for _ in range(reps_at_each_level[1]):
            self.level3.append(EESP(out_channel_map[2], out_channel_map[2], stride=1, k=K[2], r_lim=recept_limit[2]))

        self.level4_0 = DownSampler(out_channel_map[2], out_channel_map[3], k=K[2], r_lim=recept_limit[2], reinf=self.input_reinforcement)
        self.level4 = nn.ModuleList()
        for i in range(reps_at_each_level[2]):
            self.level4.append(EESP(out_channel_map[3], out_channel_map[3], stride=1, k=K[3], r_lim=recept_limit[3]))

        self.level5_0 = DownSampler(out_channel_map[3], out_channel_map[4], k=K[3], r_lim=recept_limit[3])
        self.level5 = nn.ModuleList()
        for i in range(reps_at_each_level[3]):
            self.level5.append(EESP(out_channel_map[4], out_channel_map[4], stride=1, k=K[4], r_lim=recept_limit[4]))

        self.level5.append(CBR(out_channel_map[4], out_channel_map[4], 3, 1, groups=out_channel_map[4]))
        self.level5.append(CBR(out_channel_map[4], out_channel_map[5], 1, 1, groups=K[4]))

        self.classifier = nn.Linear(out_channel_map[5], num_classes)
        self.config = out_channel_map
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input, p=0.2):
        out_l1 = self.level1(input)
        if not self.input_reinforcement:
            del input
            input = None

        out_l2 = self.level2_0(out_l1, input)

        out_l3_0 = self.level3_0(out_l2, input)
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        out_l4_0 = self.level4_0(out_l3, input)
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)

        out_l5_0 = self.level5_0(out_l4)
        for i, layer in enumerate(self.level5):
            if i == 0:
                out_l5 = layer(out_l5_0)
            else:
                out_l5 = layer(out_l5)

        output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)
        output_g = F.dropout(output_g, p=p, training=self.training)
        output_1x1 = output_g.view(output_g.size(0), -1)

        return self.classifier(output_1x1)

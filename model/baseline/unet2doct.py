import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

from model.block.octconv import *
from model.block.sk import *


class UNet2d(nn.Module):
    def __init__(self, n_channels, n_classes, pretrained=False):
        super(UNet2d, self).__init__()
        self.inc = InConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.mid_outc = OutConv(512, 512)
        self.sk = ResDilatedPyramid(512, 512)
        self.mid_inc = InConv(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        x1_h, x1_l = self.inc(x)
        x2_h, x2_l = self.down1((x1_h, x1_l))
        x3_h, x3_l = self.down2((x2_h, x2_l))
        x4_h, x4_l = self.down3((x3_h, x3_l))
        x5_h, x5_l = self.down4((x4_h, x4_l))
        mid_x_out = self.mid_outc((x5_h, x5_l))
        sk = self.sk(mid_x_out[0])
        mid_x_h_in, mid_x_l_in = self.mid_inc(sk)
        x_h, x_l = self.up1((mid_x_h_in, mid_x_l_in), (x4_h, x4_l))
        x_h, x_l = self.up2((x_h, x_l), (x3_h, x3_l))
        x_h, x_l = self.up3((x_h, x_l), (x2_h, x2_l))
        x_h, x_l = self.up4((x_h, x_l), (x1_h, x1_l))
        x = self.outc((x_h, x_l))
        return x[0]

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain = torch.load('./run/segthor/journal/experiment_3/checkpoint.pth.tar')
        self.load_state_dict(pretrain['state_dict'])


class DoubleConv(nn.Module):
    '''
        (conv => BN => ReLU) * 2
    '''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv1 = Conv_BN_ACT(in_ch, out_ch, 3, padding=1)
        self.conv2 = Conv_BN_ACT(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))

        return x_h, x_l


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv1 = Conv_BN_ACT(in_ch, out_ch, 3, padding=1, alpha_in=0, alpha_out=0.25)
        self.conv2 = Conv_BN_ACT(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))
        return x_h, x_l


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x_h, x_l = self.mp(x[0]), self.mp(x[1])
        x_h, x_l = self.conv((x_h, x_l))
        return x_h, x_l


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        # normal
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1_h, x1_l = self.up(x1[0]), self.up(x2[1])
        x2_h, x2_l = x2

        # input is CHW
        diffY_h = x2_h.size()[2] - x1_h.size()[2]
        diffX_h = x2_h.size()[3] - x1_h.size()[3]

        x1_h = F.pad(x1_h, (diffX_h // 2, diffX_h - diffX_h // 2,
                            diffY_h // 2, diffY_h - diffY_h // 2))

        diffY_l = x2_l.size()[2] - x1_l.size()[2]
        diffX_l = x2_l.size()[3] - x1_l.size()[3]

        x1_l = F.pad(x1_l, (diffX_l // 2, diffX_l - diffX_l // 2,
                            diffY_l // 2, diffY_l - diffY_l // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x_h = torch.cat([x2_h, x1_h], dim=1)
        x_l = torch.cat([x2_l, x1_l], dim=1)
        x_h, x_l = self.conv((x_h, x_l))
        return x_h, x_l


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = OctaveConv(in_ch, out_ch, 1, alpha_in=0.25, alpha_out=0)

    def forward(self, x):
        x = self.conv(x)
        return x

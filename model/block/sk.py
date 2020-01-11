import math
import torch
from torch import nn


class SKConv(nn.Module):
    def __init__(self, in_ch, dilations, r=2, l=32):
        super(SKConv, self).__init__()
        d = max(int(in_ch/r), l)

        self.convs = nn.ModuleList([])
        for i in range(len(dilations)):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=dilations[i], dilation=dilations[i], stride=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU()
            ))
        # s -> z
        self.fc = nn.Linear(in_ch, d)
        # z -> a, b, c,...
        self.fcs = nn.ModuleList([])
        for i in range(len(dilations)):
            self.fcs.append(
                nn.Linear(d, in_ch)
            )
        self.softmax = nn.Softmax(1)
        
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze(1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], 1)
        U = torch.sum(feas, 1)
        # U -> s
        s = U.mean(-1).mean(-1)
        #s = self.relu(s)
        # s -> z
        z = self.fc(s)
        # z -> a, b, c...
        for i, fc in enumerate(self.fcs):
            vector = fc(z).unsqueeze(1)
            if i == 0:
                energy_vectors = vector
            else:
                energy_vectors = torch.cat([energy_vectors, vector], 1)
        energy_vectors = self.softmax(energy_vectors)
        energy_vectors = energy_vectors.unsqueeze(-1).unsqueeze(-1)
        # s * (a, b, c...)
        v = (feas * energy_vectors).sum(1)
        return v


class ResDilatedPyramid(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResDilatedPyramid, self).__init__()
        dilations = [1, 2, 4, 8]
        inter_ch = out_ch // 2

        self.sk = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, 1, stride=1),
            nn.BatchNorm2d(inter_ch),
            SKConv(inter_ch, dilations=dilations, r=2),
            nn.BatchNorm2d(inter_ch),
            nn.Conv2d(inter_ch, out_ch, 1, stride=1),
            nn.BatchNorm2d(out_ch)
        )

        if in_ch == out_ch:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=1),
                nn.BatchNorm2d(out_ch)
            )
            
        self._init_weight()
            
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        fea = self.sk(x)
        return fea + self.shortcut(x)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def to_cuda(self, var):
        if self.cuda:
            var = var.cuda()
        return var

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal' or 'dice' or 'tversky']"""
        if mode == 'ce':
            return self.cross_entropy_loss
        elif mode == 'focal':
            return self.focal_loss
        elif mode == 'dice':
            return self.dice_loss
        elif mode == 'tversky':
            return self.tversky_loss
        else:
            raise NotImplementedError

    def cross_entropy_loss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def focal_loss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def dice_loss(self, logit, target, smooth=1e-10):
        logit = F.softmax(logit)
        n, c, h, w = logit.size()

        target = target.unsqueeze(1)
        target_onehot = self.to_cuda(torch.zeros(n, c, h, w))
        target_onehot.scatter_(1, target.long(), 1)
        target_onehot.requires_grad = True

        idc = [1, 2, 3, 4]
        pc = logit[:, idc, ...].type(torch.float32)
        tc = target_onehot[:, idc, ...].type(torch.float32)
        intersection = torch.einsum('bchw,bchw->bc', pc, tc)
        union = torch.einsum('bchw->bc', pc) + torch.einsum('bchw->bc', tc)
        divided = 1 - (2 * intersection + smooth) / (union + smooth)
        loss = torch.sum(divided.mean(1), 0)

        if self.batch_average:
            loss /= n

        return loss

    def tversky_loss(self, logit, target, alpha=0.3):
        logit = F.softmax(logit)
        n, c, h, w = logit.size()

        loss = 0
        
        target = target.unsqueeze(1)
        target_onehot = self.to_cuda(torch.zeros(n, c, h, w))
        target_onehot.scatter_(1, target.long(), 1)
        target_onehot.requires_grad = True

        for i in range(c):
            pred = logit[:, i, ...]
            gt = target_onehot[:, i, ...]
            beta = 1 - alpha
            tp = (pred * gt).sum()
            fp = ((1 - gt) * pred).sum()
            fn = (gt * (1 - pred)).sum()
            tversky = 1 - tp / (tp + alpha*fp + beta*fn)
            loss += torch.sum(tversky, 0)

        if self.batch_average:
            loss /= n

        return loss

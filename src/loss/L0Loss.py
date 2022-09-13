import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.modules.loss._Loss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, sr, hr):
        sr_ = (sr != 0).sum().float()
        hr_ = (hr != 0).sum().float()

        l = F.l1_loss(sr_, hr_)

        return l
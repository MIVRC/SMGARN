import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResUnit(nn.Module):
    def __init__(self, dim):
        super(ResUnit, self).__init__()

        self.act = nn.ReLU(True)
        self.conv1 = default_conv(dim, dim, 3)
        self.conv2 = default_conv(dim, dim*2, 1)
        self.conv3 = default_conv(dim*2, dim, 1)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)

        return x + shortcut


class FusionBlock(nn.Module):
    def __init__(self, n_color, embed_dim):
        super(FusionBlock, self).__init__()

        self.act = nn.ReLU(True)

        self.conv_1 = default_conv(n_color, embed_dim, 3)
        self.conv_2 = default_conv(embed_dim, embed_dim, 3)

        self.conv_1_2 = default_conv(embed_dim, embed_dim, 3)
        self.conv_2_2 = default_conv(embed_dim, embed_dim, 3)

        self.ru_1 = ResUnit(embed_dim)
        self.ru_2 = ResUnit(embed_dim)

        self.ru_1_1 = ResUnit(embed_dim)
        self.ru_2_1 = ResUnit(embed_dim)

        self.ru = ResUnit(embed_dim)
        self.ru_ = ResUnit(embed_dim)

        self.conv_tail_1 = default_conv(embed_dim*2, embed_dim, 3)
        self.conv_tail_2 = default_conv(embed_dim, embed_dim, 3)

    def forward(self, img_snow, mask):

        img_snow = self.ru_1(self.conv_1(img_snow))
        mask = self.ru_2(self.conv_2(mask))

        img_1 = self.ru(self.conv_1_2((img_snow-mask)))

        img_snow = self.ru_1_1(img_snow)
        mask = self.ru_2_1(mask)

        img_2 = self.ru_(self.conv_2_2((img_snow-mask)))


        return self.conv_tail_2(self.act(self.conv_tail_1(torch.cat((img_1, img_2), dim=1))))


class MARB(nn.Module):
    def __init__(self, dim):
        super(MARB, self).__init__()

        self.act = nn.ReLU(True)

        self.conv_dl2 = default_conv(dim, dim, 1)
        self.conv_dl3 = default_conv(dim, dim, 3)
        self.conv_dl5 = default_conv(dim, dim, 5)

        self.conv1_1 = default_conv(dim, dim, 3)
        self.conv1_2 = default_conv(dim, dim, 3)
        self.conv1_3 = default_conv(dim, dim, 3)

        self.conv2_1 = default_conv(dim*2, dim, 3)
        self.conv2_2 = default_conv(dim*2, dim, 3)

        self.conv_tail = default_conv(dim*2, dim, 3)

    def forward(self, x):
        x1 = self.conv1_1(self.conv_dl2(x))
        x2 = self.conv1_2(self.conv_dl3(x))
        x3 = self.conv1_3(self.conv_dl5(x))

        x_cat_1 = self.conv2_1(torch.cat((x1, x2), dim=1))
        x_cat_2 = self.conv2_2(torch.cat((x2, x3), dim=1))

        return self.conv_tail(self.act(torch.cat((x_cat_1, x_cat_2), dim=1))) + x

# class MARB(nn.Module):
#     def __init__(self, dim):
#         super(MARB, self).__init__()
#
#         self.act = nn.ReLU(True)
#
#         self.conv_dl2 = default_conv(dim, dim, 1)
#         self.conv_dl3 = default_conv(dim, dim, 3)
#         self.conv_dl5 = default_conv(dim, dim, 5)
#
#         self.conv1_1 = default_conv(dim, dim, 3)
#         self.conv1_2 = default_conv(dim, dim, 3)
#         self.conv1_3 = default_conv(dim, dim, 3)
#
#         # self.conv2_1 = default_conv(dim*2, dim, 3)
#         # self.conv2_2 = default_conv(dim*2, dim, 3)
#
#         self.conv_tail = default_conv(dim*3, dim, 3)
#
#     def forward(self, x):
#         x1 = self.conv1_1(self.conv_dl2(x))
#         x2 = self.conv1_2(self.conv_dl3(x))
#         x3 = self.conv1_3(self.conv_dl5(x))
#
#         # x_cat_1 = self.conv2_1(torch.cat((x1, x2), dim=1))
#         # x_cat_2 = self.conv2_2(torch.cat((x2, x3), dim=1))
#
#         return self.conv_tail(self.act(torch.cat((x1, x2, x3), dim=1))) + x


class MaskBlock(nn.Module):
    def __init__(self, embed_dim):
        super(MaskBlock, self).__init__()
        self.act = nn.ReLU(True)
        self.conv_head = default_conv(embed_dim, embed_dim, 3)

        self.conv_self = default_conv(embed_dim, embed_dim, 1)

        self.conv1 = default_conv(embed_dim, embed_dim, 3)
        self.conv1_1 = default_conv(embed_dim, embed_dim, 1)
        self.conv1_2 = default_conv(embed_dim, embed_dim, 1)
        self.conv_tail = default_conv(embed_dim, embed_dim, 3)

    def forward(self, x):
        x = self.conv_head(x)
        x = self.conv_self(x)
        x = x.mul(x)
        x = self.act(self.conv1(x))
        x = self.conv1_1(x).mul(self.conv1_2(x))

        return self.conv_tail(x)
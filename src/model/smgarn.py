from model import common
# from Train.model import common
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return SMGARN(args)


class SnowMaskBlock(nn.Module):
    def __init__(self, embed_dim):
        super(SnowMaskBlock, self).__init__()
        self.smblock = common.MaskBlock(embed_dim)
        self.conv3 = common.default_conv(embed_dim, embed_dim, 3)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        shortcut = x
        x = self.smblock(x)
        x = self.norm(x.flatten(2).transpose(-1, -2))
        x = self.conv3(x.transpose(-1, -2).view(B, -1, H, W))
        return x + shortcut


class Mask_Net(nn.Module):
    def __init__(self, n_colors, embed_dim, conv):
        super(Mask_Net, self).__init__()
        h = []
        h.append(conv(n_colors, embed_dim, 3))
        h.append(conv(embed_dim, embed_dim, 3))
        self.head = nn.Sequential(*h)
        self.g_mp1 = SnowMaskBlock(embed_dim)

        self.conv_out1 = common.default_conv(embed_dim, embed_dim, 3)
        self.conv_out2 = common.default_conv(embed_dim, 3, 3)

    def forward(self, x):
        x = self.head(x)
        out_1 = self.g_mp1(x)
        out_1 = self.conv_out1(out_1)

        out = self.conv_out2(out_1)
        return out, out_1

class ReconstructNet(nn.Module):
    def __init__(self, n_colors, dim, depth):
        super(ReconstructNet, self).__init__()
        self.fusion = common.FusionBlock(n_colors, dim)
        block = []
        for i in range(depth):
            block.append(common.MARB(dim))
        self.recon = nn.Sequential(*block)
        t = []
        t.append(common.default_conv(dim, dim, 3))
        t.append(nn.ReLU(True))
        t.append(common.default_conv(dim, n_colors, 3))
        self.tail = nn.Sequential(*t)

    def forward(self, x, mask):
        x = self.fusion(x, mask)
        out = self.recon(x) + x
        out = self.tail(out)
        return out



class SMGARN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SMGARN, self).__init__()
        print("SMGARN")
        n_colors = 3
        dim = 112
        ReconNet_num = 3

        self.Stage_I = Mask_Net(n_colors=n_colors, embed_dim=dim, conv=conv)

        self.Stage_II = ReconstructNet(n_colors, dim, ReconNet_num)

    def forward(self, x):
        mask, mask_feature = self.Stage_I(x)
        x = self.Stage_II(x, mask_feature)
        return x, mask
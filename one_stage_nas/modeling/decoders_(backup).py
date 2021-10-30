import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .common import conv1x1_bn, conv3x3_bn, sep3x3_bn


class ASPPModule(nn.Module):
    """ASPP module of DeepLab V3+. Using separable atrous conv.
    Currently no GAP. Don't think GAP is useful for cityscapes.
    """

    def __init__(self, inp, oup, rates, affine=True, use_gap=True):
        super(ASPPModule, self).__init__()
        self.conv1 = conv1x1_bn(inp, oup, 1, affine=affine)
        self.atrous = nn.ModuleList()
        self.use_gap = use_gap
        for rate in rates:
            self.atrous.append(sep3x3_bn(inp, oup, rate))
        num_branches = 1 + len(rates)
        if use_gap:
            self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     conv1x1_bn(inp, oup, 1))
            num_branches += 1
        self.conv_last = conv1x1_bn(oup * num_branches,
                                    oup, 1, affine=affine)

    def forward(self, x):
        atrous_outs = [atrous(x) for atrous in self.atrous]
        atrous_outs.append(self.conv1(x))
        if self.use_gap:
            gap = self.gap(x)
            gap = F.interpolate(gap, size=x.size()[2:],
                                mode='bilinear', align_corners=False)
            atrous_outs.append(gap)
        x = torch.cat(atrous_outs, dim=1)
        x = self.conv_last(x)
        return x


class DeepLabDecoder(nn.Module):
    """DeepLab V3+ decoder
    """

    def __init__(self, cfg, out_stride):
        super(DeepLabDecoder, self).__init__()
        BxF = cfg.MODEL.NUM_BLOCKS * cfg.MODEL.FILTER_MULTIPLIER
        inp = BxF * out_stride
        rates = cfg.MODEL.ASPP_RATES
        self.aspp = ASPPModule(inp, 256, rates, use_gap=False)
        self.proj = conv1x1_bn(BxF, 48, 1)
        self.conv0 = sep3x3_bn(304, 256)
        self.conv1 = sep3x3_bn(256, 256)
        self.clf = nn.Conv2d(256, cfg.MODEL.IN_CHANNEL, 1, padding=0)

    def forward(self, x, targets=None, loss_dict=None, loss_weight=None):
        x0, x1 = x
        x1 = self.aspp(x1)
        x1 = F.interpolate(x1, size=x0.size()[2:], mode='bilinear',
                           align_corners=False)
        x = torch.cat((self.proj(x0), x1), dim=1)
        x = self.conv1(self.conv0(x))
        pred = self.clf(x)
        if self.training:
            if loss_dict is not None:
                loss = []
                for loss_item, weight in zip(loss_dict, loss_weight):
                    loss.append(loss_item(pred, targets) * weight)
            else:
                loss = F.mse_loss(pred, targets)
            return pred, loss

        else:
            return pred, {}


class AutoDeepLabDecoder(nn.Module):
    """ ASPP Module at each output features
    """

    def __init__(self, cfg, out_strides):
        super(AutoDeepLabDecoder, self).__init__()
        self.aspps = nn.ModuleList()
        BxF = cfg.MODEL.NUM_BLOCKS * cfg.MODEL.FILTER_MULTIPLIER
        affine = cfg.MODEL.AFFINE
        num_strides = len(out_strides)
        for i, out_stride in enumerate(out_strides):
            if cfg.MODEL.META_MODE == "Scale":
                rate = 32 // out_stride
                inp = int(BxF * out_stride // 4)
            elif cfg.MODEL.META_MODE == "Width":
                rate = out_stride
                inp = int(BxF * np.power(2, i))

            oup = BxF
            self.aspps.append(ASPPModule(inp, oup, [rate],
                                         affine=affine,
                                         use_gap=False))
        self.pre_cls = conv3x3_bn(BxF * num_strides,
                                  BxF * num_strides,
                                  1, affine=affine)
        # self.clf = nn.Conv2d(BxF * num_strides,
        #                      cfg.DATASET.NUM_CHANNEL, 3, padding=1)
        self.clf = nn.Sequential(
            nn.Conv2d(BxF * num_strides,
                      cfg.MODEL.IN_CHANNEL, 3, padding=1),
            nn.Sigmoid()
        )



    def forward(self, x, targets=None, loss_dict=None, loss_weight=None):
        l1_size = x[0].size()[2:]
        x = [aspp(x_i) for aspp, x_i in zip(self.aspps, x)]
        x = [F.interpolate(x_i, size=l1_size, mode='bilinear') if i > 0 else x_i
             for i, x_i in enumerate(x)]
        x = self.pre_cls(torch.cat(x, dim=1))
        pred = self.clf(x)

        if self.training:
            if loss_dict is not None:
                loss = []
                for loss_item, weight in zip(loss_dict, loss_weight):
                    loss.append(loss_item(pred, targets) * weight)
            else:
                loss = F.mse_loss(pred, targets)
            return pred, loss

        else:
            return pred, {}


def build_decoder(cfg, out_strides=[4, 8, 16, 32]):
    """
    out_stride (int or List)
    """
    if cfg.SEARCH.SEARCH_ON:
        if cfg.MODEL.META_MODE == "Scale":
            out_strides = np.logspace(2, cfg.MODEL.NUM_STRIDES + 1, cfg.MODEL.NUM_STRIDES, base=2, dtype=np.int16)
        elif cfg.MODEL.META_MODE == "Width":
            out_strides = np.ones(cfg.MODEL.NUM_STRIDES, np.int16) * 16
        return AutoDeepLabDecoder(cfg, out_strides)
    else:
        return DeepLabDecoder(cfg, out_strides)

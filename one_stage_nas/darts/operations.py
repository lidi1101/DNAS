"""
DARTS operations
"""
import torch.nn as nn
from DCNv2.dcn_v2 import DCN
from decompositions.decompositions import cp_decomposition_conv_layer,tucker_decomposition_conv_layer


OPS = {
    'none' : lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'con_conv_3x3' : lambda C, stride, affine: ReLUConvBN(C, C, 3, stride, 1, dilation=1, affine=affine),
    'con_conv_5x5' : lambda C, stride, affine: ReLUConvBN(C, C, 5, stride, 2, dilation=1, affine=affine),
    'skip_connect' : lambda C, stride, affine: Identity(),
    'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3' : lambda C, stride, affine: ReLUConvBN(C, C, 3, stride, 2, dilation=2, affine=affine),
    'dil_conv_5x5' : lambda C, stride, affine: ReLUConvBN(C, C, 5, stride, 4, dilation=2, affine=affine),
    'dil_3x3_2' : lambda C, stride, affine: SepConv(C, C, 3, stride, 2, dilation=2, affine=affine),
    'def_3x3' : lambda C, stride, affine: DefConv(C, C, 3, affine=affine),
    'def_5x5': lambda C, stride, affine: DefConv(C, C, 5, affine=affine),
    'dil_3x3_4' : lambda C, stride, affine: SepConv(C, C, 3, stride, 4, dilation=4, affine=affine),
    'dil_3x3_8' : lambda C, stride, affine: SepConv(C, C, 3, stride, 8, dilation=8, affine=affine),
    'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
        ),
}


class ReLUConvBN(nn.Module):
    """not used"""

    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 dilation=1, affine=True):
        super(ReLUConvBN, self).__init__()
        # if C_in==15:
        #     tempconv=tucker_decomposition_conv_layer(nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
        #             padding=padding, dilation=dilation, groups=C_in,
        #             bias=False),[5,5])
        # else:
        #     tempconv=nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
        #             padding=padding, dilation=dilation, groups=C_in,
        #             bias=False)
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            # tempconv,
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1,
                 affine=True, repeats=2):
        super(SepConv, self).__init__()
        # if C_in==C_out==15:
        #     tempconv=tucker_decomposition_conv_layer(nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
        #             padding=padding, dilation=dilation, groups=C_in,
        #             bias=False),[8,8])
        # else:
        #     tempconv=nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
        #             padding=padding, dilation=dilation, groups=C_in,
        #             bias=False)
        basic_op = lambda: nn.Sequential(
          nn.ReLU(inplace=False),
          nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=C_in,
                    bias=False),
          # tempconv,
          nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
          nn.BatchNorm2d(C_in, affine=affine),
        )
        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx),
                               basic_op())

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)


class DefConv(nn.Module):

    def __init__(self, C_in, C_out, ksize, affine=True):
        super(DefConv, self).__init__()
        self.dcn = nn.Sequential(nn.ReLU(inplace=False),
                                 DCN(C_in, C_out, ksize, stride=1,
                                     padding=ksize // 2, deformable_groups=2),
                                 nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.dcn(x)

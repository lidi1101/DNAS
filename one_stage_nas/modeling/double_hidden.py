"""
Implements Auto-DeepLab framework
"""

import torch
from torch import nn
import torch.nn.functional as F

from one_stage_nas.darts.cell import Cell
from one_stage_nas.darts.genotypes import PRIMITIVES
from .decoders import build_decoder
from .common import conv3x3_bn, conv1x1_bn, viterbi


class Router(nn.Module):
    """ Propagate hidden states to next layer

    """

    def __init__(self, ind, inp, C, num_strides=4, affine=True):
        """
        Arguments:
            ind (int) [2-5]: index of the cell, which decides output scales
            inp (int): inp size
            C (int): output size of the same scale
        """
        super(Router, self).__init__()
        self.ind = ind
        self.num_strides = num_strides

        if ind > 0:
            # upsample
            self.postprocess0 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                conv1x1_bn(inp, C // 2, 1, affine=affine, relu=False))
        self.postprocess1 = conv3x3_bn(inp, C, 1, affine=affine, relu=False)
        if ind < num_strides - 1:
            # downsample
            self.postprocess2 = conv3x3_bn(inp, C * 2, 2, affine=affine, relu=False)

    def forward(self, out):
        """
        Returns:
            h_next ([Tensor]): None for empty
        """
        if self.ind > 0:
            h_next_0 = self.postprocess0(out)
        else:
            h_next_0 = None
        h_next_1 = self.postprocess1(out)
        if self.ind < self.num_strides - 1:
            h_next_2 = self.postprocess2(out)
        else:
            h_next_2 = None
        return h_next_0, h_next_1, h_next_2


class AutoDeepLab(nn.Module):
    """
    Main class for Auto-DeepLab.

    Use one cell per hidden states
    """

    def __init__(self, cfg):
        super(AutoDeepLab, self).__init__()
        self.f = cfg.MODEL.FILTER_MULTIPLIER
        self.num_layers = cfg.MODEL.NUM_LAYERS
        self.num_blocks = cfg.MODEL.NUM_BLOCKS
        self.num_strides = cfg.MODEL.NUM_STRIDES
        affine = cfg.MODEL.AFFINE
        self.stem = nn.Sequential(
            conv3x3_bn(3, 64, 2, affine=affine),
            conv3x3_bn(64, 64, 1, affine=affine),
            conv3x3_bn(64, self.f * 4 * self.num_blocks, 2, affine=affine,
                       relu=False))
        self.reduce1 = conv3x3_bn(self.f * 4 * self.num_blocks,
                                  self.f * 4, 1, affine=affine,
                                  relu=False)

        self.cells = nn.ModuleList()
        self.routers = nn.ModuleList()
        self.cell_configs = []
        self.tie_cell = cfg.SEARCH.TIE_CELL

        for l in range(1, self.num_layers + 1):
            for h in range(min(self.num_strides, l + 1)):
                stride = 2 ** (h + 2)
                C = self.f * stride

                if h < l:
                    self.routers.append(Router(h, C * self.num_blocks,
                                               C, affine=affine))

                empty_H1 = (h >= l) or (l == 1)
                self.cell_configs.append(
                    "L{}H{}: {}\t{}".format(
                        l, h, C, empty_H1))
                self.cells.append(Cell(self.num_blocks, C,
                                       empty_H1, affine=affine))

        # ASPP
        self.decoder = build_decoder(cfg)
        self.init_alphas()

    def w_parameters(self):
        return [value for key, value in self.named_parameters()
                if 'arch' not in key and value.requires_grad]

    def a_parameters(self):
        a_params = [value for key, value in self.named_parameters() if 'arch' in key]
        return a_params

    def init_alphas(self):
        k = sum(2 + i for i in range(self.num_blocks))
        num_ops = len(PRIMITIVES)
        if self.tie_cell:
            self.arch_alphas = nn.Parameter(torch.ones(k, num_ops))
        else:
            self.arch_alphas = nn.Parameter(torch.ones(len(self.cells), k, num_ops))

        m = sum(min(l+1, self.num_strides) for l in range(self.num_layers))
        beta_weights = torch.ones(m, 3)
        # mask out
        top_inds = []
        btm_inds = []
        start = 0
        for l in range(self.num_layers):
            top_inds.append(start)
            if l+1 < self.num_strides:
                start += l+1
            else:
                start += self.num_strides
                btm_inds.append(start-1)

        beta_weights[top_inds, 0] = -50
        beta_weights[btm_inds, 2] = -50
        self.arch_betas = nn.Parameter(beta_weights)

        gamma_weights = torch.ones()
        self.score_func = F.softmax

    def scores(self):
        return (self.score_func(self.arch_alphas, dim=-1),
                self.score_func(self.arch_betas, dim=-1))

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Tensor]): ground-truth segmentation mask (optional)

        Returns:
            result (list[Tensor]): the output from the model.
                During training, it returns a Tensor which contains the loss.
                During testing, it returns list[Tensor] contains the mask.
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed.")

        alphas, betas = self.scores()

        # The first layer is different
        features = self.stem(images)
        hidden_states = [features]
        inputs_1 = [self.reduce1(features)]

        cell_ind = 0
        router_ind = 0
        for l in range(self.num_layers):
            # prepare next inputs
            inputs_0 = [0] * min(l + 2, self.num_strides)
            for i, hs in enumerate(hidden_states):
                # print('router {}: '.format(router_ind), self.cell_configs[router_ind])
                h_0, h_1, h_2 = self.routers[router_ind](hs)
                # print(h_0 is None, h_1 is None, h_2 is None)
                # print(betas[router_ind])
                if i > 0:
                    inputs_0[i-1] = inputs_0[i-1] + h_0 * betas[router_ind][0]
                inputs_0[i] = inputs_0[i] + h_1 * betas[router_ind][1]
                if i < self.num_strides-1:
                    inputs_0[i+1] = inputs_0[i+1] + h_2 * betas[router_ind][2]
                router_ind += 1

            # run cells
            hidden_states = []
            for i, s0 in enumerate(inputs_0):
                # prepare next input
                if i < len(inputs_1):
                    s1 = inputs_1[i]
                else:
                    s1 = 0
                # print('cell: ', self.cell_configs[cell_ind])
                if self.tie_cell:
                    cell_weights = alphas
                else:
                    cell_weights = alphas[cell_ind]
                hidden_states.append(self.cells[cell_ind](s0, s1, cell_weights))
                cell_ind += 1

            inputs_1 = inputs_0

        # apply ASPP on hidden_state
        pred, loss = self.decoder(hidden_states, targets)

        if self.training:
            return {'decoder_loss': loss}
        else:
            return pred

    def get_path_genotype(self, betas):
        # construct transition matrix
        trans = []
        b_ind = 0
        for l in range(self.num_layers):
            layer = []
            for i in range(self.num_strides):
                if i < l + 1:
                    layer.append(betas[b_ind].detach().numpy().tolist())
                    b_ind += 1
                else:
                    layer.append([0, 0, 0])
            trans.append(layer)
        return viterbi(trans)

    def genotype(self):
        alphas, betas = self.scores()
        if self.tie_cell:
            gene_cell = self.cells[0].genotype(alphas)
        else:
            gene_cell = []
            for i, cell in enumerate(self.cells):
                gene_cell.append(cell.genotype(alphas[i]))
        gene_path = self.get_path_genotype(betas)
        return gene_cell, gene_path



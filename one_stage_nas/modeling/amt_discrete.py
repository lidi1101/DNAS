"""
Discrete structure of Auto-DeepLab

Includes utils to convert continous Auto-DeepLab to discrete ones
"""

import os
import torch
from torch import nn
from torch.nn import functional as F

from one_stage_nas.darts.cell import FixCell
from .auto_multitask import AutoMultiTask
from .common import conv3x3_bn, conv1x1_bn
from .decoders import build_decoder
from .loss import loss_dict



def get_genotype_from_adl(cfg):
    # create ADL model
    adl_cfg = cfg.clone()
    adl_cfg.defrost()
    # modified by hkzhang in 2019-5-9
    # adl_cfg.merge_from_list(['MODEL.META_ARCHITECTURE', 'AutoDeepLab',
    #                          'MODEL.FILTER_MULTIPLIER', 8,
    #                          'MODEL.AFFINE', True])

    adl_cfg.merge_from_list(['MODEL.META_ARCHITECTURE', 'AutoDeepLab',
                             'MODEL.FILTER_MULTIPLIER', 8,
                             'MODEL.AFFINE', True,
                             'SEARCH.SEARCH_ON', True])

    model = AutoMultiTask(adl_cfg)
    # load weights
    SEARCH_RESULT_DIR = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                              '{}_{}'.format(cfg.DATASET.DATA_NAME, cfg.RESULT_DIR),
                              'search/models/model_best.pth'))
    # ckpt = torch.load(cfg.MODEL.SEARCH_RESULT)
    ckpt = torch.load(SEARCH_RESULT_DIR)
    restore = {k: v for k, v in ckpt['model'].items() if 'arch' in k}
    model.load_state_dict(restore, strict=False)
    return model.genotype()


class Scaler(nn.Module):
    """Reshape features"""
    def __init__(self, scale, inp, C):
        """
        Arguments:
            scale (int) [-2, 2]: scale < 0 for downsample
        """
        super(Scaler, self).__init__()
        if scale == 0:
            self.scale = conv1x1_bn(inp, C, 1, relu=False)
        if scale == 1:
            self.scale = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=False),
                conv1x1_bn(inp, C, 1, relu=False))
        if scale == 2:
            self.scale = nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear',
                            align_corners=False),
                conv1x1_bn(inp, C, 1, relu=False))
        # official implementation used bilinear for all scalers
        if scale == -1:
            self.scale = conv3x3_bn(inp, C, 2, relu=False)
        if scale == -2:
            self.scale = nn.Sequential(conv3x3_bn(inp, inp * 2, 2),
                                       conv3x3_bn(inp * 2, C, 2, relu=False))

    def forward(self, hidden_state):
        return self.scale(hidden_state)


class DeepLabScaler_Scale(nn.Module):
    """Official implementation
    https://github.com/tensorflow/models/blob/master/research/deeplab/core/nas_cell.py#L90
    """
    def __init__(self, scale, inp, C):
        super(DeepLabScaler_Scale, self).__init__()
        self.scale = 2 ** scale
        self.conv = conv1x1_bn(inp, C, 1, relu=False)

    def forward(self, hidden_state):
        if self.scale != 1:
            hidden_state = F.interpolate(hidden_state,
                                         scale_factor=self.scale,
                                         mode='bilinear',
                                         align_corners=False)
        return self.conv(F.relu(hidden_state))


class DeepLabScaler_Width(nn.Module):
    """Official implementation
    https://github.com/tensorflow/models/blob/master/research/deeplab/core/nas_cell.py#L90
    """
    def __init__(self, scale, inp, C):
        super(DeepLabScaler_Width, self).__init__()
        self.scale = 2 ** scale
        self.conv = conv1x1_bn(inp, C, 1, relu=False)

    def forward(self, hidden_state):
        return self.conv(F.relu(hidden_state))


class AMTDiscrete(nn.Module):
    def __init__(self, cfg):
        super(AMTDiscrete, self).__init__()

        # load genotype
        if len(cfg.DATASET.TRAIN_DATASETS) == 0:
            geno_file = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                  '{}_{}'.format(cfg.DATASET.DATA_NAME, cfg.RESULT_DIR),
                                  'search/models/model_best.geno'))
        else:
            geno_file = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                  '{}_{}'.format('&'.join(cfg.DATASET.TRAIN_DATASETS), cfg.RESULT_DIR),
                                  'search/models/model_best.geno'))
        # geno_file = cfg.MODEL.GENOTYPE
        if os.path.exists(geno_file):
            print("Loading genotype from {}".format(geno_file))
            genotype = torch.load(geno_file, map_location=torch.device("cpu"))
        else:
            genotype = get_genotype_from_adl(cfg)
            # print(genotype)
            # save genotype
            print("Saving genotype to {}".format(geno_file))
            torch.save(genotype, geno_file)

        geno_cell, geno_path = genotype

        # modified the architecture manually to test does the architecture searched by NAS is really better than others
        # geno_cell_denoise_new_1 = [('def_3x3', 0), ('def_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1), ('skip_connect', 0), ('def_3x3', 2), ('def_3x3', 4), ('def_3x3', 0), ('skip_connect', 5), ('skip_connect', 1)]
        # genotype_denoise_new1 = []
        # genotype_denoise_new1.append(geno_cell_denoise_new_1)
        # genotype_denoise_new1.append(geno_path)
        #
        # geno_cell_denoise_new_2 = [('con_conv_3x3', 0), ('con_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1), ('skip_connect', 0), ('def_3x3', 2), ('def_3x3', 3), ('def_3x3', 1), ('skip_connect', 5), ('skip_connect', 1)]
        # genotype_denoise_new2 = []
        # genotype_denoise_new2.append(geno_cell_denoise_new_2)
        # genotype_denoise_new2.append(geno_path)

        # geno_cell_derain_new_1 = [('def_3x3', 1), ('skip_connect', 0), ('def_3x3', 0), ('def_3x3', 1), ('def_3x3', 2), ('skip_connect', 0), ('def_3x3', 2), ('def_3x3', 1)]
        # genotype_derain_new1 = []
        # genotype_derain_new1.append(geno_cell_derain_new_1)
        # genotype_derain_new1.append(geno_path)
        #
        # geno_cell_derain_new_2 = [('def_3x3', 1), ('skip_connect', 0), ('def_3x3', 0), ('con_conv_3x3', 1), ('def_3x3', 2), ('skip_connect', 0), ('def_3x3', 4), ('con_conv_3x3', 0)]
        # genotype_derain_new2 = []
        # genotype_derain_new2.append(geno_cell_derain_new_2)
        # genotype_derain_new2.append(geno_path)



        self.genotpe = genotype

        if 0 in geno_path:
            self.endpoint = (len(geno_path) - 1) - list(reversed(geno_path)).index(0)
        else:
            self.endpoint = None

        # basic configs
        self.f = cfg.MODEL.FILTER_MULTIPLIER
        self.num_layers = cfg.MODEL.NUM_LAYERS
        self.num_blocks = cfg.MODEL.NUM_BLOCKS
        self.num_strides = cfg.MODEL.NUM_STRIDES
        self.in_channel = cfg.MODEL.IN_CHANNEL
        self.stem1 = nn.Sequential(
            # nn.BatchNorm2d(self.in_channel),
            conv3x3_bn(self.in_channel, 64, 1),
            # conv3x3_bn(64, 64, 1, relu=False)
            )
        self.stem2 = conv3x3_bn(64, 64, 1, relu=False)
        self.reduce = conv3x3_bn(64, self.f*self.num_blocks, 1, affine=False, relu=False)

        # create cells
        self.cells = nn.ModuleList()
        self.scalers = nn.ModuleList()
        if cfg.SEARCH.TIE_CELL:
            geno_cell = [geno_cell] * self.num_layers


        if cfg.MODEL.META_MODE == "Scale":
            DeepLabScaler = DeepLabScaler_Scale
        elif cfg.MODEL.META_MODE == "Width":
            DeepLabScaler = DeepLabScaler_Width

        h_0 = 0  # prev hidden index
        h_1 = -1  # prev prev hidden index
        for layer, (geno, h) in enumerate(zip(geno_cell, geno_path), 1):
            stride = 2 ** h
            self.cells.append(FixCell(geno, self.f * stride))
            # scalers
            if layer == 1:
                inp0 = 64
                inp1 = 64
            elif layer == 2:
                inp0 = 2 ** h_0 * self.f * self.num_blocks
                inp1 = 64
            else:
                inp0 = 2 ** h_0 * self.f * self.num_blocks
                inp1 = 2 ** h_1 * self.f * self.num_blocks

            if layer == 1:
                scaler0 = DeepLabScaler(h_0 - h, inp0,
                                        stride * self.f)
                scaler1 = DeepLabScaler(h_0 - h, inp1,
                                        stride * self.f)
            else:
                scaler0 = DeepLabScaler(h_0 - h, inp0,
                                        stride * self.f)
                scaler1 = DeepLabScaler(h_1 - h, inp1,
                                        stride * self.f)

            h_1 = h_0
            h_0 = h
            self.scalers.append(scaler0)
            self.scalers.append(scaler1)
        self.decoder = build_decoder(cfg, out_strides=stride)
        if cfg.SOLVER.LOSS is not None:
            self.loss_dict = []
            self.loss_weight = []
            for loss_item, loss_weight in zip(cfg.SOLVER.LOSS, cfg.SOLVER.LOSS_WEIGHT):
                if 'ssim' in loss_item or 'grad' in loss_item:
                    self.loss_dict.append(loss_dict[loss_item](channel=cfg.MODEL.IN_CHANNEL))
                else:
                    self.loss_dict.append(loss_dict[loss_item]())
                self.loss_weight.append(loss_weight)

        else:
            self.loss_dict = None
            self.loss_weight = None

    def genotype(self):
        return self.genotpe

    def forward(self, images, targets=None, drop_prob=-1):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed.")

        h1 = self.stem1(images)
        h0 = self.stem2(F.relu(h1))

        if self.endpoint==None:
            endpoint = self.reduce(h0)

        for i, cell in enumerate(self.cells):
            s0 = self.scalers[i*2](h0)
            s1 = self.scalers[i*2+1](h1)
            h1 = h0
            h0 = cell(s0, s1, drop_prob)
            if self.endpoint is not None and i == self.endpoint:
                endpoint = h0

        pred, loss = self.decoder([endpoint, F.relu(h0, inplace=True)], targets, self.loss_dict, self.loss_weight)

        if self.training:
            return pred, {'decoder_loss': sum(loss)/len(loss)}
        else:
            return pred


class AMTDiscrete_geno(nn.Module):
    def __init__(self, args, cfg):
        super(AMTDiscrete_geno, self).__init__()

        # geno_file = cfg.MODEL.GENOTYPE
        geno_file = args.geno_file
        if os.path.exists(geno_file):
            print("Loading genotype from {}".format(geno_file))
            genotype = torch.load(geno_file, map_location=torch.device("cpu"))
        else:
            print('geno_file "{}" does not exist'.format(geno_file))

        geno_cell, geno_path = genotype

        self.genotpe = genotype

        if 0 in geno_path:
            self.endpoint = (len(geno_path) - 1) - list(reversed(geno_path)).index(0)
        else:
            self.endpoint = None

        # basic configs
        self.f = cfg.MODEL.FILTER_MULTIPLIER
        self.num_layers = cfg.MODEL.NUM_LAYERS
        self.num_blocks = cfg.MODEL.NUM_BLOCKS
        self.num_strides = cfg.MODEL.NUM_STRIDES
        self.in_channel = cfg.MODEL.IN_CHANNEL
        self.stem1 = nn.Sequential(
            # nn.BatchNorm2d(self.in_channel),
            conv3x3_bn(self.in_channel, 64, 1),
            # conv3x3_bn(64, 64, 1, relu=False)
            )
        self.stem2 = conv3x3_bn(64, 64, 1, relu=False)
        self.reduce = conv3x3_bn(64, self.f*self.num_blocks, 1, affine=False, relu=False)

        # create cells
        self.cells = nn.ModuleList()
        self.scalers = nn.ModuleList()
        if cfg.SEARCH.TIE_CELL:
            geno_cell = [geno_cell] * self.num_layers


        if cfg.MODEL.META_MODE == "Scale":
            DeepLabScaler = DeepLabScaler_Scale
        elif cfg.MODEL.META_MODE == "Width":
            DeepLabScaler = DeepLabScaler_Width

        h_0 = 0  # prev hidden index
        h_1 = -1  # prev prev hidden index
        for layer, (geno, h) in enumerate(zip(geno_cell, geno_path), 1):
            stride = 2 ** h
            self.cells.append(FixCell(geno, self.f * stride))
            # scalers
            if layer == 1:
                inp0 = 64
                inp1 = 64
            elif layer == 2:
                inp0 = 2 ** h_0 * self.f * self.num_blocks
                inp1 = 64
            else:
                inp0 = 2 ** h_0 * self.f * self.num_blocks
                inp1 = 2 ** h_1 * self.f * self.num_blocks

            if layer == 1:
                scaler0 = DeepLabScaler(h_0 - h, inp0,
                                        stride * self.f)
                scaler1 = DeepLabScaler(h_0 - h, inp1,
                                        stride * self.f)
            else:
                scaler0 = DeepLabScaler(h_0 - h, inp0,
                                        stride * self.f)
                scaler1 = DeepLabScaler(h_1 - h, inp1,
                                        stride * self.f)

            h_1 = h_0
            h_0 = h
            self.scalers.append(scaler0)
            self.scalers.append(scaler1)
        self.decoder = build_decoder(cfg, out_strides=stride)
        # if cfg.SOLVER.LOSS is not None:
        #     self.loss_dict = []
        #     self.loss_weight = []
        #     for loss_item, loss_weight in zip(cfg.SOLVER.LOSS, cfg.SOLVER.LOSS_WEIGHT):
        #         if 'ssim' in loss_item or 'grad' in loss_item:
        #             self.loss_dict.append(loss_dict[loss_item](channel=cfg.MODEL.IN_CHANNEL))
        #         else:
        #             self.loss_dict.append(loss_dict[loss_item]())
        #         self.loss_weight.append(loss_weight)


        self.loss_dict = None
        self.loss_weight = None

    def genotype(self):
        return self.genotpe

    def forward(self, images, targets=None, drop_prob=-1):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed.")

        h1 = self.stem1(images)
        h0 = self.stem2(F.relu(h1))

        if self.endpoint==None:
            endpoint = self.reduce(h0)

        for i, cell in enumerate(self.cells):
            s0 = self.scalers[i*2](h0)
            s1 = self.scalers[i*2+1](h1)
            h1 = h0
            h0 = cell(s0, s1, drop_prob)
            if self.endpoint is not None and i == self.endpoint:
                endpoint = h0

        pred, loss = self.decoder([endpoint, F.relu(h0, inplace=True)], targets, self.loss_dict, self.loss_weight)

        if self.training:
            return pred, {'decoder_loss': sum(loss)/len(loss)}
        else:
            return pred

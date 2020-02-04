import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.costvolume_regularization import CostVolumeComputation
from models.feature_extraction import UnaryExtraction


class OmniMVS(nn.Module):
    def __init__(self, sweep, ndisp, min_depth, w, h):
        super(OmniMVS, self).__init__()

        self.cam_list = ['cam1', 'cam2', 'cam3', 'cam4']
        self.ndisp = ndisp
        self.min_depth = min_depth
        self.w = w
        self.h = h

        # sweeping sphere inverse distance
        self.inv_depths = np.linspace(0, 1 / min_depth, ndisp) + np.finfo(np.float32).eps
        self.depths = 1. / self.inv_depths

        # module
        self.feature_extraction = UnaryExtraction()
        self.sweep = sweep
        self.transference = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(
            nn.Conv3d(32 * len(self.cam_list), 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.cost_regularization = CostVolumeComputation()

    def forward(self, batch):
        # randomly permuted concatenate
        cam_idxs = list(range(len(self.cam_list)))
        if self.training:
            random.shuffle(cam_idxs)
        device = batch[self.cam_list[0]].device
        dtype = batch[self.cam_list[0]].dtype
        batch_size = batch[self.cam_list[0]].size(0)
        # assert batch_size == 1, "batch_size should be one"

        # define emply cost volume
        costs = torch.zeros((batch_size, 32 * len(self.cam_list), self.ndisp // 2, self.h // 2, self.w // 2),
                            device=device, dtype=dtype)

        # construct cost volume
        for cam_idx in cam_idxs:
            key = self.cam_list[cam_idx]
            # feature extraction
            feat = self.feature_extraction(batch[key])

            # spherical sweeping
            warps = torch.zeros((batch_size, 32, self.ndisp // 2, self.h, self.w), device=device, dtype=dtype)
            for d_idx, d in enumerate(self.depths[::2]):
                # warp feature
                grid = self.sweep.get_grid(cam_idx, d)
                grid = grid.type_as(feat)
                grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
                warps[:, :, d_idx:d_idx + 1, :, :] = F.grid_sample(feat, grid, align_corners=False).unsqueeze(2)
            warps = self.transference(warps)
            costs[:, 32 * cam_idx:32 * (cam_idx + 1), :, :, :] = warps

        # fusion
        costs = self.fusion(costs)

        # cost volume computation
        out = self.cost_regularization(costs)

        # disparity
        pred_disp = DisparityRegression(self.ndisp)(out)
        return pred_disp


class DisparityRegression(nn.Module):
    """ Soft argmax disparity regression proposed in [1]

    Parameters
    ----------
    ndisp : int
        Number of disparity,
    min_disp : int
        Minimum disparity. Usually disparity starts from zero.

    References
    ----------
    [1] A. Kendall et al., “End-to-end learning of geometry and context for deep stereo regression”
    """

    def __init__(self, ndisp, min_disp=0):
        super(DisparityRegression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(min_disp, ndisp + min_disp)), [1, ndisp, 1, 1]))

    def forward(self, x):
        x = F.softmax(torch.squeeze(x, 1), dim=1)
        self.disp = self.disp.to(x.device)
        self.disp.requires_grad_(False)
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out

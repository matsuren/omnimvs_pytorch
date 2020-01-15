import numpy as np
import torch


class InvDepthConverter(object):
    def __init__(self, ndisp, invd_0, invd_max):
        self._ndisp = ndisp
        self._invd_0 = invd_0
        self._invd_max = invd_max

    def invdepth_to_index(self, idepth, round_value=True):
        invd_idx = (self._ndisp - 1) * (idepth - self._invd_0) / (self._invd_max - self._invd_0)
        # Q: why round?
        if round_value:
            invd_idx = torch.round(invd_idx)
        return invd_idx

    def index_to_invdepth(self, invd_idx):
        idepth = self._invd_0 + invd_idx * (self._invd_max - self._invd_0) / (self._ndisp - 1)
        return idepth


def evaluation_metrics(preds, gts, ndisp, crop=True):
    """Evaluation metrics for OmniMVS

    Parameters
    ----------
    preds : torch.tensor
        Prediction invdepth index torch tensor [B x H x W]
    gts : torch.tensor
        Groundtruth invdepth index torch tensor [B x H x W]
    ndisp : int
        Number of disparity
    crop : bool
        Crop area e (-np.pi/4 < theta < np.pi/4)　(ignore  polar region)

    Returns
    -------
        total_errors : numpy array
            error values of ["a1", "a3", "a5", "mae", "rms"]
        error_names : list of string
            ["a1", "a3", "a5", "mae", "rms"]
    """
    if isinstance(preds, list):
        preds = torch.cat(preds)
    if isinstance(gts, list):
        gts = torch.cat(gts)
    assert preds.size() == gts.size()
    assert preds.ndim == 3

    pred_array = preds.detach().cpu().numpy()
    gt_array = gts.detach().cpu().numpy()
    b, h, w = pred_array.shape

    if crop:
        pred_array = pred_array[:, h // 4:h // 4 + h // 2, :]
        gt_array = gt_array[:, h // 4:h // 4 + h // 2, :]

    # calculate total mean error
    total_errors = []
    for one_pred, one_gt in zip(pred_array, gt_array):
        error = np.abs(one_pred - one_gt).flatten()
        # Eq.(3) error
        error = (error / ndisp) * 100

        a1 = (error > 1).mean() * 100  # %
        a3 = (error > 3).mean() * 100  # %
        a5 = (error > 5).mean() * 100  # %

        mae = error.mean()
        rms = np.sqrt((error ** 2).mean())

        total_errors.append([a1, a3, a5, mae, rms])

    total_errors = np.array(total_errors).mean(axis=0)
    error_names = ["a1", "a3", "a5", "mae", "rms"]
    return total_errors, error_names

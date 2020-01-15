import torch


class InvDepthConverter(object):
    def __init__(self, ndisp, invd_0, invd_max):
        self._ndisp = ndisp
        self._invd_0 = invd_0
        self._invd_max = invd_max

    def invdepth_to_index(self, idepth):
        invd_idx = (self._ndisp - 1) * (idepth - self._invd_0) / (self._invd_max - self._invd_0)
        # Q: why round?
        invd_idx = torch.round(invd_idx)
        return invd_idx

    def index_to_invdepth(self, invd_idx):
        idepth = self._invd_0 + invd_idx * (self._invd_max - self._invd_0) / (self._ndisp - 1)
        return idepth

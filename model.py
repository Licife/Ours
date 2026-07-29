import torch
import torch.optim
import torch.nn as nn

import config as c
from hinet import Hinet
from invblock import ALM
import modules.Unet_common as common


class Model_1(nn.Module):
    """Stage 1 model for the 1-th secret image in a cascade."""

    def __init__(self):
        super(Model_1, self).__init__()

        # Each stage owns an independent Hinet and an independent ALM.
        self.model = Hinet()
        self.alm = ALM()

        self.iwt = common.IWT()
        self.dwt = common.DWT()

    def forward(self, x, rev=False):
        if not rev:
            # x = cat(carrier_dwt, secret_dwt), 24 channels.
            out = self.model(x)
            stego_freq, lost_info_freq = torch.chunk(out, 2, dim=1)

            # ALM works in the spatial domain.
            stego_spatial = self.iwt(stego_freq)
            z_spatial = self.alm(stego_spatial)
            z_pred_freq = self.dwt(z_spatial)

            # Keep the original project interface:
            # out = cat(stego_freq, lost_info_freq), z_pred_freq from ALM.
            return out, z_pred_freq

        # x is the current-stage stego image in the wavelet domain.
        stego_spatial = self.iwt(x)
        z_spatial = self.alm(stego_spatial)
        z_freq = self.dwt(z_spatial)

        # Reverse the current stage to recover its carrier and secret image.
        rev_in = torch.cat((x, z_freq), dim=1)
        out = self.model(rev_in, rev=True)
        return out


class Model_2(nn.Module):
    """Stage 2 model for the 2-th secret image in a cascade."""

    def __init__(self):
        super(Model_2, self).__init__()

        # Each stage owns an independent Hinet and an independent ALM.
        self.model = Hinet()
        self.alm = ALM()

        self.iwt = common.IWT()
        self.dwt = common.DWT()

    def forward(self, x, rev=False):
        if not rev:
            # x = cat(carrier_dwt, secret_dwt), 24 channels.
            out = self.model(x)
            stego_freq, lost_info_freq = torch.chunk(out, 2, dim=1)

            # ALM works in the spatial domain.
            stego_spatial = self.iwt(stego_freq)
            z_spatial = self.alm(stego_spatial)
            z_pred_freq = self.dwt(z_spatial)

            # Keep the original project interface:
            # out = cat(stego_freq, lost_info_freq), z_pred_freq from ALM.
            return out, z_pred_freq

        # x is the current-stage stego image in the wavelet domain.
        stego_spatial = self.iwt(x)
        z_spatial = self.alm(stego_spatial)
        z_freq = self.dwt(z_spatial)

        # Reverse the current stage to recover its carrier and secret image.
        rev_in = torch.cat((x, z_freq), dim=1)
        out = self.model(rev_in, rev=True)
        return out


class Model_3(nn.Module):
    """Stage 3 model for the 3-th secret image in a cascade."""

    def __init__(self):
        super(Model_3, self).__init__()

        # Each stage owns an independent Hinet and an independent ALM.
        self.model = Hinet()
        self.alm = ALM()

        self.iwt = common.IWT()
        self.dwt = common.DWT()

    def forward(self, x, rev=False):
        if not rev:
            # x = cat(carrier_dwt, secret_dwt), 24 channels.
            out = self.model(x)
            stego_freq, lost_info_freq = torch.chunk(out, 2, dim=1)

            # ALM works in the spatial domain.
            stego_spatial = self.iwt(stego_freq)
            z_spatial = self.alm(stego_spatial)
            z_pred_freq = self.dwt(z_spatial)

            # Keep the original project interface:
            # out = cat(stego_freq, lost_info_freq), z_pred_freq from ALM.
            return out, z_pred_freq

        # x is the current-stage stego image in the wavelet domain.
        stego_spatial = self.iwt(x)
        z_spatial = self.alm(stego_spatial)
        z_freq = self.dwt(z_spatial)

        # Reverse the current stage to recover its carrier and secret image.
        rev_in = torch.cat((x, z_freq), dim=1)
        out = self.model(rev_in, rev=True)
        return out


class Model_4(nn.Module):
    """Stage 4 model for the 4-th secret image in a cascade."""

    def __init__(self):
        super(Model_4, self).__init__()

        # Each stage owns an independent Hinet and an independent ALM.
        self.model = Hinet()
        self.alm = ALM()

        self.iwt = common.IWT()
        self.dwt = common.DWT()

    def forward(self, x, rev=False):
        if not rev:
            # x = cat(carrier_dwt, secret_dwt), 24 channels.
            out = self.model(x)
            stego_freq, lost_info_freq = torch.chunk(out, 2, dim=1)

            # ALM works in the spatial domain.
            stego_spatial = self.iwt(stego_freq)
            z_spatial = self.alm(stego_spatial)
            z_pred_freq = self.dwt(z_spatial)

            # Keep the original project interface:
            # out = cat(stego_freq, lost_info_freq), z_pred_freq from ALM.
            return out, z_pred_freq

        # x is the current-stage stego image in the wavelet domain.
        stego_spatial = self.iwt(x)
        z_spatial = self.alm(stego_spatial)
        z_freq = self.dwt(z_spatial)

        # Reverse the current stage to recover its carrier and secret image.
        rev_in = torch.cat((x, z_freq), dim=1)
        out = self.model(rev_in, rev=True)
        return out


class Model_5(nn.Module):
    """Stage 5 model for the 5-th secret image in a cascade."""

    def __init__(self):
        super(Model_5, self).__init__()

        # Each stage owns an independent Hinet and an independent ALM.
        self.model = Hinet()
        self.alm = ALM()

        self.iwt = common.IWT()
        self.dwt = common.DWT()

    def forward(self, x, rev=False):
        if not rev:
            # x = cat(carrier_dwt, secret_dwt), 24 channels.
            out = self.model(x)
            stego_freq, lost_info_freq = torch.chunk(out, 2, dim=1)

            # ALM works in the spatial domain.
            stego_spatial = self.iwt(stego_freq)
            z_spatial = self.alm(stego_spatial)
            z_pred_freq = self.dwt(z_spatial)

            # Keep the original project interface:
            # out = cat(stego_freq, lost_info_freq), z_pred_freq from ALM.
            return out, z_pred_freq

        # x is the current-stage stego image in the wavelet domain.
        stego_spatial = self.iwt(x)
        z_spatial = self.alm(stego_spatial)
        z_freq = self.dwt(z_spatial)

        # Reverse the current stage to recover its carrier and secret image.
        rev_in = torch.cat((x, z_freq), dim=1)
        out = self.model(rev_in, rev=True)
        return out


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5' or split[-2] == 'conv2':
                param.data.fill_(0.)

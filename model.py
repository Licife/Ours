import torch.optim
import torch.nn as nn
import config as c
from hinet import Hinet
from invblock import *
import modules.Unet_common as common

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = Hinet()
        self.alm = ALM()

        self.iwt = common.IWT()
        self.dwt = common.DWT()  # Spatial -> Freq (3 -> 12)

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

            stego_freq, lost_info_freq = torch.chunk(out, 2, dim=1)

            # 2. 频域转空间域 (给 ALM 看)
            stego_spatial = self.iwt(stego_freq)

            # 3. ALM 生成辅助变量 (空间域)
            z_spatial = self.alm(stego_spatial)

            # 4. 【关键步骤】把 ALM 的输出转回频域
            # 这样 z_pred 的尺寸 (12, 112, 112) 才能和 lost_info_freq (12, 112, 112) 计算 Loss
            z_pred_freq = self.dwt(z_spatial)

            return out, z_pred_freq

        else:
            stego_spatial = self.iwt(x)

            # 2. ALM 生成辅助变量 (空间域)
            z_spatial = self.alm(stego_spatial)

            # 3. 【关键步骤】把 z 转回频域 (以便与 x 拼接)
            z_freq = self.dwt(z_spatial)  # [B, 12, 112, 112]

            # 4. 拼接 (现在两者尺寸都是 112x112 了)
            rev_in = torch.cat((x, z_freq), dim=1)

            # 5. 反向恢复
            out = self.model(rev_in, rev=True)

            return out

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#
#         self.model = Hinet()
#
#         # ALM has been removed
#         self.iwt = common.IWT()
#         self.dwt = common.DWT()  # Spatial -> Freq (3 -> 12)
#
#     def forward(self, x, rev=False):
#
#         if not rev:
#             out = self.model(x)
#             # Without ALM, we just return the raw INN output
#             return out
#
#         else:
#             # x is the output_steg (frequency domain)
#             # Since ALM is removed, we sample standard gaussian noise for the lost information (z)
#             z_freq = torch.randn_like(x)
#
#             # Concatenate stego and noise to restore the required input dimension
#             rev_in = torch.cat((x, z_freq), dim=1)
#
#             # Reverse recovery
#             out = self.model(rev_in, rev=True)
#
#             return out

def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5' or split[-2] == 'conv2':
                param.data.fill_(0.)

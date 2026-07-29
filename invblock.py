from math import exp
import config as c
from rrdb_denselayer import *

import torch.nn.functional as F
import torch
import torch.nn as nn

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SCA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=True)
        )

    def forward(self, x):
        return x * self.sca(x)

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = SCA(dw_channel // 2)
        self.sg = SimpleGate()
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x

class SpatialAttention(nn.Module):
    """
    CBAM 的空间注意力模块 (Spatial Attention)
    沿通道维度计算均值和最大值，拼接后经过 7x7 卷积学习空间权重。
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 接收通道拼接后的 2 个通道输入
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度 (dim=1) 分别计算均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接成 2 个通道 -> [B, 2, H, W]
        pool_out = torch.cat([avg_out, max_out], dim=1)

        # 卷积降维到 1 个通道并经过 Sigmoid 归一化
        scale = self.sigmoid(self.conv(pool_out))
        return x * scale

class ALM(nn.Module):
    """
    Deterministic Dual-Branch Information Compensation Module.

    The module contains two parts:

    1) Extractor E
       - Three encoder stages: Conv4x4(stride=2) + NAFBlock.
       - Spatial attention is applied at all three scales.
       - The first two attended encoder features are used as additive skip features.
       - The deepest attended feature enters the decoder.
       - The first two transposed-convolution layers are followed by NAFBlock.
       - The final transposed convolution produces F_SI.

    2) Deterministic Generator G
       - Upper branch: four Conv3x3 + NAFBlock units and one Conv1x1,
         followed by a long residual connection to produce beta_z.
       - Lower branch: two Conv3x3 + LeakyReLU units and one linear
         Conv1x1 layer to produce the signed detail correction delta_z.
       - No Gaussian sampling or stochastic modulation is used.
       - The final auxiliary variable is z = beta_z + delta_z.

    Default tensor shapes:
        x_st     : [B, 3, H, W]
        F_SI     : [B, 3, H, W]
        beta_z   : [B, 3, H, W]
        delta_z  : [B, 3, H, W]
        z        : [B, 3, H, W]

    H and W should normally be divisible by 8 because the extractor performs
    three 2x downsampling operations.
    """

    def __init__(self):
        super(ALM, self).__init__()

        # ================================================================
        # Part I: Extractor E
        # ================================================================
        # Encoder stage 1:
        # [B, 3, H, W] -> [B, 64, H/2, W/2] -> same shape after NAFBlock.
        self.down1 = nn.Conv2d(
            3, 64, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.enc1 = NAFBlock(c=64)

        # Encoder stage 2:
        # [B, 64, H/2, W/2] -> [B, 128, H/4, W/4] -> same shape.
        self.down2 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.enc2 = NAFBlock(c=128)

        # Encoder stage 3:
        # [B, 128, H/4, W/4] -> [B, 256, H/8, W/8] -> same shape.
        self.down3 = nn.Conv2d(
            128, 256, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.enc3 = NAFBlock(c=256)

        # Spatial attention at all three encoder scales.
        # Attention changes feature weights but does not change tensor size.
        self.sa_skip1 = SpatialAttention(kernel_size=7)  # 64 channels, H/2
        self.sa_skip2 = SpatialAttention(kernel_size=7)  # 128 channels, H/4
        self.sa_deep = SpatialAttention(kernel_size=7)   # 256 channels, H/8

        # Decoder stage 1:
        # [B, 256, H/8, W/8] -> [B, 128, H/4, W/4] -> NAF refinement.
        self.up1 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.dec1 = NAFBlock(c=128)

        # Decoder stage 2:
        # [B, 128, H/4, W/4] -> [B, 64, H/2, W/2] -> NAF refinement.
        self.up2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.dec2 = NAFBlock(c=64)

        # Extractor output:
        # [B, 64, H/2, W/2] -> [B, 3, H, W] = F_SI.
        self.up3 = nn.ConvTranspose2d(
            64, 3, kernel_size=4, stride=2, padding=1
        )

        # ================================================================
        # Part II: Generator G - upper structural compensation branch
        # ================================================================
        # Unit 1: Conv3x3 3->64, then NAFBlock; spatial size remains H x W.
        self.beta_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.beta_naf1 = NAFBlock(c=64)

        # Units 2-4: Conv3x3 64->64, then NAFBlock; size remains unchanged.
        self.beta_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.beta_naf2 = NAFBlock(c=64)

        self.beta_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.beta_naf3 = NAFBlock(c=64)

        self.beta_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.beta_naf4 = NAFBlock(c=64)

        # Conv1x1 maps 64 channels back to 3 channels.
        # Its output is added to F_SI through the long residual connection.
        self.beta_out = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)

        # ================================================================
        # Part II: Generator G - lower deterministic detail branch
        # ================================================================
        # This lightweight branch directly predicts a signed residual
        # correction delta_z from F_SI. The final layer is linear: there is
        # no Sigmoid because compensation values must be able to be positive
        # or negative.
        #
        # [B, 3, H, W] -> [B, 32, H, W]
        self.delta_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.delta_act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # [B, 32, H, W] -> [B, 16, H, W]
        self.delta_conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.delta_act2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # [B, 16, H, W] -> [B, 3, H, W] = delta_z
        self.delta_out = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)

        # Start the lower branch from zero contribution. At initialization,
        # z is therefore approximately beta_z; the lightweight branch then
        # gradually learns only the extra local detail that is still missing.
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)

    def extract_condition_feature(self, x_st):
        """Extract stego-conditioned compensation feature F_SI."""
        # ---------------- Encoder ----------------
        fd1 = self.enc1(self.down1(x_st))       # [B, 64, H/2, W/2]
        fd2 = self.enc2(self.down2(fd1))        # [B, 128, H/4, W/4]
        fd3 = self.enc3(self.down3(fd2))        # [B, 256, H/8, W/8]

        # Spatial attention at all three scales; tensor sizes are unchanged.
        fd1_attended = self.sa_skip1(fd1)
        fd2_attended = self.sa_skip2(fd2)
        fd3_attended = self.sa_deep(fd3)

        # ---------------- Decoder ----------------
        fu1 = self.up1(fd3_attended, output_size=fd2.shape)
        fu1 = self.dec1(fu1)                    # [B, 128, H/4, W/4]
        fu1 = fu1 + fd2_attended

        fu2 = self.up2(fu1, output_size=fd1.shape)
        fu2 = self.dec2(fu2)                    # [B, 64, H/2, W/2]
        fu2 = fu2 + fd1_attended

        f_si = self.up3(fu2, output_size=x_st.shape)  # [B, 3, H, W]
        return f_si

    def generate_auxiliary_variable(self, f_si):
        """
        Generate the deterministic auxiliary variable.

        beta_z  = F_SI + G_beta(F_SI)
        delta_z = G_delta(F_SI)
        z       = beta_z + delta_z
        """
        # ---------------- Upper branch: beta_z ----------------
        beta_feat = self.beta_naf1(self.beta_conv1(f_si))
        beta_feat = self.beta_naf2(self.beta_conv2(beta_feat))
        beta_feat = self.beta_naf3(self.beta_conv3(beta_feat))
        beta_feat = self.beta_naf4(self.beta_conv4(beta_feat))

        beta_residual = self.beta_out(beta_feat)  # [B, 3, H, W]
        beta_z = f_si + beta_residual              # long residual addition

        # ---------------- Lower branch: delta_z ----------------
        delta_feat = self.delta_act1(self.delta_conv1(f_si))
        delta_feat = self.delta_act2(self.delta_conv2(delta_feat))
        delta_z = self.delta_out(delta_feat)       # signed linear correction

        # ---------------- Final deterministic compensation ----------------
        z_spatial = beta_z + delta_z
        return z_spatial, beta_z, delta_z

    def forward(self, x_st, return_params=False):
        """
        Args:
            x_st: Stego image with shape [B, 3, H, W].
            return_params: When True, also return F_SI and both branch outputs.

        Returns:
            Default:
                z_spatial
            When return_params=True:
                z_spatial, F_SI, beta_z, delta_z
        """
        if x_st.ndim != 4 or x_st.size(1) != 3:
            raise ValueError(
                f'x_st must have shape [B, 3, H, W], got {tuple(x_st.shape)}.'
            )

        f_si = self.extract_condition_feature(x_st)
        z_spatial, beta_z, delta_z = self.generate_auxiliary_variable(f_si)

        if return_params:
            return z_spatial, f_si, beta_z, delta_z
        return z_spatial

class INV_block(nn.Module):
    def __init__(self, subnet_constructor_1=ResidualDenseBlock_out_CE, subnet_constructor_2=ResidualDenseBlock_out_SE,clamp=c.clamp, harr=True, in_1=3, in_2=3):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp
        # ρ
        self.r = subnet_constructor_2(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor_2(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor_1(self.split_len1, self.split_len2)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:

            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1


        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)

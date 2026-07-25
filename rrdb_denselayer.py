import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import modules.module_util as mutil
from timm.layers import DropPath
import config as c

# CA + SA
class ChannelAttention_CBAM(nn.Module):
    """
    CBAM 的通道注意力模块 (Channel Attention)
    利用全局平均池化和全局最大池化，通过共享的 MLP 网络学习通道权重。
    """

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention_CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的 MLP (使用 1x1 卷积实现)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        # 将平均池化和最大池化的结果相加，然后经过 Sigmoid 归一化
        scale = self.sigmoid(avg_out + max_out)
        return x * scale

class SpatialAttention_CBAM(nn.Module):
    """
    CBAM 的空间注意力模块 (Spatial Attention)
    沿通道维度计算均值和最大值，拼接后经过 7x7 卷积学习空间权重。
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention_CBAM, self).__init__()
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

# Dense connection
class ResidualDenseBlock_out_CE(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out_CE, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.attention = ChannelAttention_CBAM(input + 4 * 32, reduction=16)
        # initialization
        mutil.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x = self.attention(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = self.conv5(x)
        return x5

# class ResidualDenseBlock_out_CE(nn.Module):
#     def __init__(self, input, output, bias=True):
#         super(ResidualDenseBlock_out_CE, self).__init__()
#         self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
#         self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
#         self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
#         self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
#         self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
#         self.lrelu = nn.LeakyReLU(inplace=True)
#         self.attention = SpatialAttention_CBAM()
#         # initialization
#         mutil.initialize_weights([self.conv5], 0.)
#
#     def forward(self, x):
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
#         x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
#         x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
#         x = self.attention(torch.cat((x, x1, x2, x3, x4), 1))
#         x5 = self.conv5(x)
#         return x5


# OSA
class ResidualDenseBlock_out_SE(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out_SE, self).__init__()

        gc = 32

        self.conv1 = nn.Conv2d(input, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * gc, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.attention = SpatialAttention_CBAM()
        # initialization
        mutil.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))
        x4 = self.lrelu(self.conv4(x3))
        x = self.attention(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = self.conv5(x)
        return x5

# class ResidualDenseBlock_out_SE(nn.Module):
#     def __init__(self, input, output, bias=True):
#         super(ResidualDenseBlock_out_SE, self).__init__()
#
#         gc = 32
#
#         self.conv1 = nn.Conv2d(input, gc, 3, 1, 1, bias=bias)
#         self.conv2 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
#         self.conv3 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
#         self.conv4 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
#         self.conv5 = nn.Conv2d(input + 4 * gc, output, 3, 1, 1, bias=bias)
#         self.lrelu = nn.LeakyReLU(inplace=True)
#         self.attention = SpatialAttention_CBAM()
#         # initialization
#         mutil.initialize_weights([self.conv5], 0.)
#
#     def forward(self, x):
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(x1))
#         x3 = self.lrelu(self.conv3(x2))
#         x4 = self.lrelu(self.conv4(x3))
#         x = self.attention(torch.cat((x, x1, x2, x3, x4), 1))
#         x5 = self.conv5(x)
#         return x5
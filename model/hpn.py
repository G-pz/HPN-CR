import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
import numpy as np
from utils.get_featuremaps import save_feature_to_img


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class Gconv(nn.Module):
    def __int__(self, dim, order=3, s=1.0):
        super().__int__()
        self.order = order
        self.scale = s
        self.dims = [dim // 2 ** i for i in range(order)].reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        self.dwconv = nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=7, padding=(7-1)//2, bias=True, groups=sum(self.dims))
        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i +1], 1) for i in range(order-1)]
        )
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]
        x = self.proj_out(x)

        return x
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = nn.Sequential()
        if in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        self.pool = nn.Sequential()
        if pool:
            self.pool = nn.Sequential(
                nn.AvgPool2d(2, 2, 0)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = residual + out
        out = self.pool(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.reduction = reduction

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()  
        x = torch.transpose(x, 1, 2)  
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  
    coords_flatten = torch.flatten(coords, 1)  
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.gate_2 = nn.Conv2d(2, 2, 3, 1, 1)
        self.gate_4 = nn.Conv2d(4, 4, 3, 1, 1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x

class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None, order=1):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        self.dwconv = nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=7, padding=(7 - 1) // 2, bias=True,
                                groups=sum(self.dims))
        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )
        self.proj_out = nn.Conv2d(dim, dim, 1)

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        if self.conv_type == 'DWConv' or self.use_attn:
            fused_x = self.proj_in(X)
            pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
            dw_abc = self.dwconv(abc)
            dw_list = torch.split(dw_abc, self.dims, dim=1)
            x = pwa * dw_list[0]
            for i in range(self.order - 1):
                x = self.pws[i](x) * dw_list[i + 1]
            V = self.proj_out(x)

        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)

            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]
           
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(shifted_QKV, self.window_size)  

            attn_windows = self.attn(qkv)

            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  

            out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
            attn_out = out.permute(0, 3, 1, 2)

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)

        else:
            if self.conv_type == 'Conv':
                out = self.conv(X)  
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))

        return out

class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=True,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.compress = ChannelPool()
        self.spatial_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=(3 - 1) // 2)
        self.spatial_5 = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=(5 - 1) // 2)
        self.spatial_7 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=(7 - 1) // 2)
        self.sa = nn.Conv2d(3, 1, 1)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        identity = x
        scale = self.compress(identity)
        spatial_attention = torch.cat([self.spatial_3(scale), self.spatial_5(scale), self.spatial_7(scale)], dim=1)
        id = identity * F.sigmoid(self.sa(spatial_attention))

        if self.use_attn: x, rescale, rebias = self.norm1(x)
        x = self.attn(x)
        if self.use_attn: x = x * rescale + rebias
        x = identity + x + 0.01*id

        identity = x
        if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm: x = x * rescale + rebias
        x = identity + x
        return x



class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class cloud_opt_sar(nn.Module):
    def __init__(self, in_chans=13, out_chans=13, window_size=[8,12,16],
                 embed_dims=[24, 48, 96, 48, 24],
                 res_dims=[64, 128, 256],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[8, 8, 8, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                 conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN]):
        super(cloud_opt_sar, self).__init__()

        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        self.SELayer1 = SELayer(channel=360, reduction=36)
        self.SELayer2 = SELayer(channel=32, reduction=8)

        self.patch_embed_9 = PatchEmbed(patch_size=2, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        self.residual1 = ResidualBlock(2, res_dims[0])
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size[0],
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        self.patch_merge1 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.residual2 = ResidualBlock(res_dims[0], res_dims[1])
        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], window_size=window_size[1],
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        self.patch_merge2 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.residual3 = ResidualBlock(res_dims[1], res_dims[2])
        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2], window_size=window_size[2],
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])
        self.residual4 = ResidualBlock(res_dims[2]+embed_dims[2], embed_dims[2], False)

        self.patch_split1 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 norm_layer=norm_layer[3], window_size=window_size[1],
                                 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.patch_split2 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                 norm_layer=norm_layer[4], window_size=window_size[0],
                                 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        self.patch_unembed_up = PatchUnEmbed(patch_size=2, out_chans=32, embed_dim=embed_dims[4], kernel_size=3)
        self.final = nn.Conv2d(in_channels=32, out_channels=out_chans, kernel_size=3, padding=1, stride=1)

        nn.init.normal_(self.skip1.weight, 0, 0.2)
        nn.init.normal_(self.skip2.weight, 0, 0.2)
        nn.init.normal_(self.final.weight, 0, 0.2)

    def forward_features(self, x, y):
        x = self.patch_embed_9(x)
        x = self.layer1(x)
        skip1 = x
        y = self.residual1(y)

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x
        y = self.residual2(y)

        x = self.patch_merge2(x)
        x = self.layer3(x)
        y = self.residual3(y)
        x = torch.cat([x, y], dim=1)
        x = self.SELayer1(x)
        x = self.residual4(x)

        x = self.patch_split1(x)
        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)

        x = self.patch_split2(x)
        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)

        x = F.leaky_relu(self.patch_unembed_up(x))
        x = self.SELayer2(x)
        x = self.final(x)

        return x

    def forward(self, x_in, y_in):
        x = self.forward_features(x_in, y_in)
        x = x + x_in
        xout = torch.sigmoid(x)

        return xout

def hpn_cr():
    return cloud_opt_sar(
        in_chans=13,
        out_chans=13,
        window_size=[8, 12, 16],
        embed_dims=[26, 52, 104, 52, 26],
        res_dims=[64, 128, 256],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 4, 2, 2],
        attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


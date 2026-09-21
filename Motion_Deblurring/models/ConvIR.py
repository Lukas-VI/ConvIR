
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class EBlock(nn.Module):
    """编码端残差块组:由 num_res 个 ResBlock 堆叠,最后一个带 filter(多尺度动态滤波)。"""
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        # 前 num_res-1 个普通残差块 + 1 个带 DeepPool 的残差块(增强特征表达)
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    """解码端残差块组:结构与 EBlock 相同,用于解码路径。"""
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    """浅层条件模块(Shallow Condition Module):在解码端每个尺度把原始下采样图像
    提炼出与当前通道数一致的浅层特征,用于给深层特征提供原始信息的引导(条件)。"""
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)  # 实例归一化,保留可学习的仿射参数
        )

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    """通道拼接融合模块(Fusion Attention Module):把深层特征与 SCM 浅层条件
    沿通道拼接后,用 1x1 卷积融合成单一特征。"""
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))  # 沿通道维拼接再卷积融合

class ConvIR(nn.Module):
    """ConvIR 主网络(去运动模糊版):编码器-解码器 U 型结构 + 多尺度监督输出。
    无版本参数,num_res 直接指定残差块个数(默认16,同主去雨副本)。"""
    def __init__(self, num_res=16):
        super(ConvIR, self).__init__()

        base_channel = 32  # 基础通道数

        # 编码器:三个尺度,通道逐级翻倍 (32->64->128)
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        # 特征提取/上下采样卷积:前3个是入口+下采样(3->32, 32->64 s2, 64->128 s2),
        # 后3个是解码端上采样到输出 (128->64 s2 trans, 64->32 s2 trans, 32->3)
        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        # 解码器:三个尺度,通道由深到浅 (128->64->32)
        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        # 深层特征与跳接/条件拼接后的降维卷积(1x1)
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        # 多尺度输出头:把各尺度解码特征投影到 3 通道图像
        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        # 融合&条件模块,分别对应不同尺度
        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)  # 1/2 分辨率输入
        x_4 = F.interpolate(x_2, scale_factor=0.5) # 1/4 分辨率输入
        z2 = self.SCM2(x_2)  # 1/2 尺度浅层条件
        z4 = self.SCM1(x_4)  # 1/4 尺度浅层条件

        outputs = list()
        # 256: 原图尺度
        x_ = self.feat_extract[0](x)      # 3 -> 32
        res1 = self.Encoder[0](x_)        # 编码器第一级
        # 128: 1/2 尺度
        z = self.feat_extract[1](res1)    # 下采样 s2 -> 64
        z = self.FAM2(z, z2)              # 融合浅层条件
        res2 = self.Encoder[1](z)         # 编码器第二级
        # 64: 1/4 尺度
        z = self.feat_extract[2](res2)    # 下采样 s2 -> 128
        z = self.FAM1(z, z4)              # 融合浅层条件
        z = self.Encoder[2](z)            # 编码器第三级(最深层)

        z = self.Decoder[0](z)            # 解码器最高尺度(1/4) -> 128
        z_ = self.ConvsOut[0](z)          # 输出 3 通道(1/4 尺度预测)
        # 128: 上采样回 1/2
        z = self.feat_extract[3](z)       # 转置卷积 s2 -> 64
        outputs.append(z_+x_4)            # 1/4 尺度预测 + 1/4 输入(残差式,监督 target 为 1/4 label)

        z = torch.cat([z, res2], dim=1)   # 与编码器第二级特征跳接 -> 128
        z = self.Convs[0](z)              # 1x1 降维 -> 64
        z = self.Decoder[1](z)            # 解码器中间尺度(1/2)
        z_ = self.ConvsOut[1](z)          # 输出 3 通道(1/2 尺度预测)
        # 256: 上采样回原图
        z = self.feat_extract[4](z)       # 转置卷积 s2 -> 32
        outputs.append(z_+x_2)            # 1/2 尺度预测 + 1/2 输入

        z = torch.cat([z, res1], dim=1)   # 与编码器第一级特征跳接 -> 64
        z = self.Convs[1](z)              # 1x1 降维 -> 32
        z = self.Decoder[2](z)            # 解码器原始尺度(原图)
        z = self.feat_extract[5](z)       # 输出 3 通道(原图尺度预测)
        outputs.append(z+x)               # 原图尺度预测 + 原图输入(残差式)

        return outputs  # 返回三个尺度:[1/4, 1/2, 原图]


def build_net():
    return ConvIR()

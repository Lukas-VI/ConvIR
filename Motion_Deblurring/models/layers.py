import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    """基础卷积单元:卷积(或转置卷积) + 可选 BatchNorm + 可选 GELU 激活。
    这是项目里最常用的积木,被几乎所有模块复用。"""
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            # 若既要用 bias 又要用 BN:BN 自带可学习的偏置,故把 conv 的 bias 关掉,避免冗余。
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            # 转置卷积(上采样)时 padding 公式略有不同,用于解码器端的分辨率恢复。
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            # 普通卷积(下采样/保持分辨率),padding 采用 same 式填充。
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            # 按需添加批归一化(需在卷积之后、激活之前)。
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            # 按需添加 GELU 激活(项目统一用 GELU,而非 ReLU)。
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    """残差块:输入经过 [3x3卷积 -> 可选DeepPool -> 3x3卷积] 后与自身相加(恒等映射)。
    (无 data 参数,同主副本结构)"""
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            DeepPoolLayer(in_channel, out_channel) if filter else nn.Identity(),  # filter 开关控制是否插入深层池化滤波
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x  # 残差连接:逐元素相加


class DeepPoolLayer(nn.Module):
    """深层多尺度池化模块:在不同空间尺度上分别池化并做动态滤波,再把多尺度信息上采样累加回原始分辨率。
    (无 data 参数,空洞率固定为 [7,9,11],同 OTS 版)"""
    def __init__(self, k, k_out):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8,4,2]   # 三种下采样的池化窗口大小
        dilation = [7,9,11]          # 每种尺度对应的空洞扩撒率(固定,动态滤波时的感受野)
        pools, convs, dynas = [],[],[]
        for j, i in enumerate(self.pools_sizes):
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))     # 平均池化,把特征压到 1/8、1/4、1/2
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))      # 尺度内卷积
            dynas.append(MultiShapeKernel(dim=k, kernel_size=3, dilation=dilation[j]))  # 动态多形状内核
        self.pools = nn.ModuleList(pools)   # 三种池化
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)    # 融合多样尺度结果的输出卷积

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                # 第一个尺度(1/8):直接池化 -> 卷积 -> 动态滤波
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                # 后续尺度(1/4、1/2):把上一尺度的输出 y_up 上采样后与当前尺度的池化结果相加,再做动态滤波
                y = self.dynas[i](self.convs[i](self.pools[i](x)+y_up))
            # 把当前尺度的结果双线性上采样回原始分辨率,累加到 resl(多尺度信息聚合)
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
                # 为下一个更大尺度准备:把当前结果放大 2 倍(逐级从 1/8 -> 1/4 -> 1/2 -> 原图)
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)        # 激活
        resl = self.conv_sum(resl)    # 输出维度对齐

        return resl

class dynamic_filter(nn.Module):
    """动态滤波器(基于数据动态生成的卷积核):由输入自身生成滤波权重,对局部 patch 做加权聚合,
    适合复原任务中内容自适应的处理。这里为方形(正方形 kernel)的动态滤波实现。"""
    def __init__(self, inchannels, kernel_size=3, dilation=1, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group        # 分组数量(动态核在通道方向分成 group 组)
        self.dilation = dilation

        # 用 1x1 卷积从特征生成动态滤波权重:输出 group*kernel_size^2 通道(每组一个 kernel)
        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Tanh()      # 滤波权重归一化到 (-1,1)
    
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        # 可学习的低频/高频门控参数(初始为0)
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)  # 低频分支权重
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)  # 高频分支权重
        self.pad = nn.ReflectionPad2d(self.dilation*(kernel_size-1)//2)          # 反射填充,保持尺寸

        self.ap = nn.AdaptiveAvgPool2d((1, 1))   # 全局平均池化 -> 生成动态核
        self.gap = nn.AdaptiveAvgPool2d(1)        # 全局平均池化 -> 提取"均值"用作低频基准

        self.inside_all = nn.Parameter(torch.zeros(inchannels,1,1), requires_grad=True)  # 可学习残差系数

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)          # 1) 全局池化,获取整图统计信息
        low_filter = self.conv(low_filter)  # 2) 动态生成滤波权重
        low_filter = self.bn(low_filter)     # 3) 归一化

        n, c, h, w = x.shape  
        # 4) 把输入按 kernel 展开成 patch(im2col),再拆成 group 组;shape: [n, group, c/group, k^2, h*w]
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        # 把动态核 reshape 成与 patch 组一致的形状,并在 group 维度插入一维用于广播
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
       
        low_filter = self.act(low_filter)  # 5) 激活到 (-1,1)
    
        # 6) 逐 patch 加权求和(等价于动态卷积),再 reshape 回 [n,c,h,w]
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        # 7) 低频路径:动态滤波结果减去一个比例的可学习"均值"基准,形成带残差的自适应结果
        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)

        out_low = out_low * self.lamb_l[None,:,None,None]   # 乘上可学习低频门控

        # 8) 高频路径:直接取输入的恒等映射乘以 (1+lamb_h)
        out_high = (identity_input) * (self.lamb_h[None,:,None,None] + 1.) 

        # 9) 低频 + 高频融合输出
        return out_low + out_high


class cubic_attention(nn.Module):
    """cubic(立体/条状)注意力:在水平(H)与垂直(W)两个方向分别做条状空间注意力,再融合。
    通过可学习 gamma/beta 做门控残差(类似 SE 或 CBAM 的门控思路)。"""
    def __init__(self, dim, group, dilation, kernel) -> None:
        super().__init__()

        # 两个方向(沿 H 的横条 与 沿 W 的竖条)的条状注意力
        self.H_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))  # 注意力输出的缩放(初始0)
        self.beta = nn.Parameter(torch.ones(dim,1,1))    # 恒等连接的门控(初始1)

    def forward(self, x):
        out = self.H_spatial_att(x)   # 先做水平条状注意力
        out = self.W_spatial_att(out) # 再做垂直条状注意力
        # 残差式融合:gamma*注意力输出 + beta*原始输入
        return self.gamma * out + x * self.beta


class spatial_strip_att(nn.Module):
    """条状空间自注意力:沿一个方向(H 或 W)做动态滤波,聚合局部一维邻域。
    H=True 为横向条(卷积核宽1高kernel),H=False 为纵向条(卷积核宽kernel高1)。"""
    def __init__(self, dim, kernel=3, dilation=1, group=2, H=True) -> None:
        super().__init__()

        self.k = kernel
        pad = dilation*(kernel-1) // 2
        self.kernel = (1, kernel) if H else (kernel, 1)   # 一维条状卷积核方向
        self.padding = (kernel//2, 1) if H else (1, kernel//2)
        self.dilation = dilation
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))  # 反射填充
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)  # 动态生成条状滤波权重
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Tanh()
        self.inside_all = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        gap_kernel = (None,1) if H else (1, None)  # 全局均值池化方向:横向条压缩垂直方向
        self.gap = nn.AdaptiveAvgPool2d(gap_kernel)

    def forward(self, x):
        identity_input = x.clone()
        filter = self.ap(x)             # 全局池化
        filter = self.conv(filter)      # 生成动态条状核
        n, c, h, w = x.shape
        # 展开邻域 patch 并按 group 分组:shape [n, group, c/group, k, h*w]
        x = F.unfold(self.pad(x), kernel_size=self.kernel, dilation=self.dilation).reshape(n, self.group, c//self.group, self.k, h*w)
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)  # 调整核形状以广播
        filter = self.filter_act(filter)
        # 动态权重加权求和(一维方向上的动态卷积),还原到原尺寸
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)

        # 同 dynamic_filter:低频残差(用方向上的均值做基准) + 高频恒等链接,均乘可学习门控
        out_low = out * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None,:,None,None]
        out_high = identity_input * (self.lamb_h[None,:,None,None]+1.)

        return out_low + out_high


class MultiShapeKernel(nn.Module):
    """多形状内核对(多形状动态滤波核):同时使用条状注意力(条形核)与方形动态滤波器(方形核),
    两者输出相加,兼顾条形感受野与方形局部细节。"""
    def __init__(self, dim, kernel_size=3, dilation=1, group=8):
        super().__init__()

        self.square_att = dynamic_filter(inchannels=dim, dilation=dilation, group=group, kernel_size=kernel_size)  # 方形动态滤波
        self.strip_att = cubic_attention(dim, group=group, dilation=dilation, kernel=kernel_size)  # 条状注意力

    def forward(self, x):

        x1 = self.strip_att(x)   # 条状分支
        x2 = self.square_att(x)  # 方形分支

        return x1+x2  # 两条分支相加
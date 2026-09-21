import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    """基础卷积单元:卷积(或转置卷积) + 可选 BatchNorm + 可选 GELU 激活。"""
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            # 用了 BN 就关闭 conv 的 bias(BN 自带偏置),避免冗余
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            # 转置卷积(上采样)的 padding 公式与普通卷积不同
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            # 普通卷积,same 式填充
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    """残差块:3x3卷积 -> (可选DeepPool) -> 3x3卷积,再与输入相加。filter=True 时嵌入多尺度动态滤波。"""
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            DeepPoolLayer(in_channel, out_channel) if filter else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class DeepPoolLayer(nn.Module):
    """深层多尺度池化:三种尺度(1/8、1/4、1/2)分别池化+动态滤波,再上采样累加回原分辨率。
    备注:此副本的 dilation 为 [7,9,11]。"""
    def __init__(self, k, k_out):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8,4,2]
        dilation = [7,9,11]   # 三个尺度用的空`洞扩撒率(与主副本 deraining 的 [3,7,9] 不同)
        pools, convs, dynas = [],[],[]
        for j, i in enumerate(self.pools_sizes):
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))     # 平均池化下采样
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
            dynas.append(MultiShapeKernel(dim=k, kernel_size=3, dilation=dilation[j]))  # 多形状动态内核
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                # 1/8 尺度:池化->卷积->动态滤波
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                # 更大尺度:上一级结果上采样后与当前池化相加再动态滤波
                y = self.dynas[i](self.convs[i](self.pools[i](x)+y_up))
            # 上采样回原分辨率并累加(多尺度信息聚合)
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl


class dynamic_filter(nn.Module):
    """动态滤波器:由输入动态生成滤波权重,对局部 patch 加权聚合,实现内容自适应的滤波。

    与主副本(deraining)的实现完全一致,详见 deraining 的 models/layers.py 逐行注释。"""
    def __init__(self, inchannels, kernel_size=3, dilation=1, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.dilation = dilation

        # 1x1 卷积从特征生成动态核权重
        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Tanh()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)  # 低频门控
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)  # 高频门控
        self.pad = nn.ReflectionPad2d(self.dilation*(kernel_size-1)//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.inside_all = nn.Parameter(torch.zeros(inchannels,1,1), requires_grad=True)  # 可学习残差系数

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)            # 全局池化 -> 生成动态核
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)

        n, c, h, w = x.shape
        # 展开成 patch 并按 group 分组
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
        low_filter = self.act(low_filter)

        # 动态加权求和(等价于动态卷积)
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        # 低频残差(减去均值基准)+ 门控
        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None,:,None,None]

        # 高频恒等路径 + 门控
        out_high = (identity_input) * (self.lamb_h[None,:,None,None] + 1.)

        return out_low + out_high


class cubic_attention(nn.Module):
    """立体条状注意力:水平(H)与垂直(W)两个方向分别做条状注意力,再用 gamma/beta 门控残差融合。"""
    def __init__(self, dim, group, dilation, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))  # 注意力的缩放(初始0)
        self.beta = nn.Parameter(torch.ones(dim,1,1))    # 恒等连接门控(初始1)

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class spatial_strip_att(nn.Module):
    """条状空间注意力:沿 H 或 W 单方向做动态滤波(如宽1高kernel / 宽kernel高1)。"""
    def __init__(self, dim, kernel=3, dilation=1, group=2, H=True) -> None:
        super().__init__()

        self.k = kernel
        pad = dilation*(kernel-1) // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)
        self.dilation = dilation
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Tanh()
        self.inside_all = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        gap_kernel = (None,1) if H else (1, None)
        self.gap = nn.AdaptiveAvgPool2d(gap_kernel)

    def forward(self, x):
        identity_input = x.clone()
        filter = self.ap(x)          # 全局池化
        filter = self.conv(filter)   # 生成条状动态核
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel, dilation=self.dilation).reshape(n, self.group, c//self.group, self.k, h*w)
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)
        filter = self.filter_act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)

        # 低频残差 + 高频恒等,各乘可学习门控
        out_low = out * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None,:,None,None]
        out_high = identity_input * (self.lamb_h[None,:,None,None]+1.)

        return out_low + out_high


class MultiShapeKernel(nn.Module):
    """多形状内核(条状 + 方形动态滤波)相加,是 ConvIR 的核心特征提取单元。"""
    def __init__(self, dim, kernel_size=3, dilation=1, group=8):
        super().__init__()

        self.square_att = dynamic_filter(inchannels=dim, dilation=dilation, group=group, kernel_size=kernel_size)
        self.strip_att = cubic_attention(dim, group=group, dilation=dilation, kernel=kernel_size)

    def forward(self, x):
        x1 = self.strip_att(x)    # 条状分支
        x2 = self.square_att(x)   # 方形分支
        return x1+x2
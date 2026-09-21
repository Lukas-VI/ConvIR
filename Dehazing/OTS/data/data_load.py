import os
import torch
import numpy as np
from PIL import Image as Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # 允许加载截断的图片(数据集部分图不完整)

def train_dataloader(path, batch_size=64, num_workers=0):
    """训练数据加载器(OTS 版):在 Dataset 内部就地完成随机裁剪(ps=256)与随机水平翻转。"""
    image_dir = os.path.join(path, 'train')

    dataloader = DataLoader(
        DeblurDataset(image_dir, ps=256),   # ps=256 表示训练时随机裁剪到 256x256
        batch_size=batch_size,
        shuffle=True,            # 训练集打乱
        num_workers=num_workers,
        pin_memory=True          # 锁页内存,加速 GPU 数据搬运
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    """测试数据加载器(OTS 版):读 'test' 子目录,is_test 以返回文件名供保存结果。"""
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    """验证数据加载器(OTS 版):同测试集目录 'test',is_valid 标记(gt 用 png)。"""
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test'), is_valid=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader

import random
class DeblurDataset(Dataset):
    """OTS 去雾成对图像数据集:在 __getitem__ 中直接做随机裁剪与随机水平翻转。
    与 ITS 版差异:训练/验证的 gt 扩展名不同(png vs jpg),增强逻辑内置而非用 Pair* 工具。"""
    def __init__(self, image_dir, transform=None, is_test=False, is_valid=False, ps=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))  # 列出所有 hazy(雾图)文件名
        self._check_image(self.image_list)   # 校验文件名后缀
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test               # 测试模式额外返回文件名
        self.is_valid = is_valid
        self.ps = ps                         # 训练随机裁剪尺寸
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx])).convert('RGB')  # 雾图
        if self.is_valid or self.is_test:      
            label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.png')).convert('RGB')  # 验证/测试 gt 为 png
        else:
            label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.jpg')).convert('RGB')  # 训练 gt 为 jpg
        ps = self.ps

        if self.ps is not None:
            # 训练增强:先转张量,再做随机同位置裁剪与随机水平翻转
            image = F.to_tensor(image)
            label = F.to_tensor(label)

            hh, ww = label.shape[1], label.shape[2]   # 图像高宽

            rr = random.randint(0, hh-ps)   # 随机裁剪左上角行
            cc = random.randint(0, ww-ps)   # 随机裁剪左上角列
            
            image = image[:, rr:rr+ps, cc:cc+ps]   # 输入与标签同一位置裁剪
            label = label[:, rr:rr+ps, cc:cc+ps]

            if random.random() < 0.5:
                image = image.flip(2)   # 随机水平翻转(翻转 W 维)
                label = label.flip(2)
        else:
            # 验证/测试:直接转张量
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        if self.is_test:
            name = self.image_list[idx]
            return image, label, name   # 测试时附带文件名,便于保存结果
        return image, label



    @staticmethod
    def _check_image(lst):
        """校验文件名后缀必须为常见图片格式,否则抛异常。"""
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError

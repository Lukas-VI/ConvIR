import os
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    """训练数据加载器(去运动模糊版):数据在 <path>/train 目录,
    子目录 blur(模糊输入)/sharp(清晰标签),使用数据增强。"""
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        # 训练增强:随机裁剪256 + 随机水平翻转 + 转张量,输入和标签同步处理
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,         # 训练集打乱顺序
        num_workers=num_workers,
        pin_memory=True       # 锁页内存加速 GPU 拷贝
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    """测试数据加载器:数据在 <path>/valid 目录,is_test=True(返回文件名)。"""
    image_dir = os.path.join(path, 'valid')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,        # 测试集不打乱
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    """验证数据加载器:同样读 <path>/valid 目录(与测试共用数据)。"""
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'valid')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    """GoPro 去运动模糊数据集:取 blur/ 目录下的模糊图作输入,
    sharp/ 目录下的清晰图(名字里 blur 换成 gt)作标签。"""
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'blur/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))  # 模糊输入
        label = Image.open(os.path.join(self.image_dir, 'sharp', self.image_list[idx].replace('blur', 'gt')))  # 清晰标签

        if self.transform:
            image, label = self.transform(image, label)   # 训练:做增强
        else:
            image = F.to_tensor(image)                    # 验证/测试:仅转张量
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name                     # 测试额外返回文件名
        return image, label

    @staticmethod
    def _check_image(lst):
        """校验文件扩展名是否为支持的图片格式之一。"""
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
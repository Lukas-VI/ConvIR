import os
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor, PairCenterCrop
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    """训练数据加载器:对输入/标签做随机裁剪、水平翻转、转张量的数据增强。"""
    image_dir = path

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256),        # 随机裁剪到 256x256
                PairRandomHorizontalFilp(), # 随机水平翻转
                PairToTensor()              # PIL -> Tensor
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,            # 训练集打乱
        num_workers=num_workers,
        pin_memory=True          # 锁页内存,加速 GPU 数据搬运
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    """测试数据加载器:默认读 'valid' 子目录,不做变换,is_test 以返回文件名。"""
    image_dir = os.path.join(path, 'valid')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    """验证数据加载器:中心裁剪到 128x128 并转张量。"""
    transform = PairCompose(
        [
            PairCenterCrop(128),
            PairToTensor()
        ]
    )
    dataloader = DataLoader(
        DeblurDataset(path, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    """成对图像数据集:目录下分为 input/ 与 target/ 两个子文件夹,文件名一一对应。"""
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'input/'))  # 列出所有输入文件名
        self._check_image(self.image_list)  # 校验文件后缀
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test              # 测试模式额外返回文件名

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'input', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'target', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)   # 成对变换(同一随机参数)
        else:
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
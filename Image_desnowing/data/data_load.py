import os
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, data='CSD', use_transform=True):
    """训练数据加载器(去雪版):读 train2500 目录,按数据集类型做随机裁剪/翻转/转张量增强。"""
    image_dir = os.path.join(path, 'train2500')

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
        DeblurDataset(image_dir, data, transform=transform),
        batch_size=batch_size,
        shuffle=True,            # 训练集打乱
        num_workers=num_workers,
        pin_memory=True          # 锁页内存,加速 GPU 数据搬运
    )
    return dataloader


def test_dataloader(path, data, batch_size=1, num_workers=0):
    """测试数据加载器(去雪版):读 test2000 目录,is_test 以返回文件名供保存结果。"""
    image_dir = os.path.join(path, 'test2000')
    dataloader = DataLoader(
        DeblurDataset(image_dir, data, is_test=True),
        
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, data, batch_size=1, num_workers=0):
    """验证数据加载器(去雪版):同测试集目录 test2000,不做变换。"""
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test2000'), data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    """去雪成对图像数据集:目录分为 Snow/(含雪图) 与 Gt/(无雪干净图),文件名一一对应。
    SRRS 数据集的 gt 需额外把扩展名改为 .jpg。"""
    def __init__(self, image_dir, data, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'Snow/'))  # 列出所有含雪图文件名
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test      # 测试模式额外返回文件名
        self.data = data
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'Snow', self.image_list[idx]))   # 含雪图
        if self.data == 'SRRS':
            # SRRS 的干净图把文件名扩展名/mid 对应为 .jpg
            label = Image.open(os.path.join(self.image_dir, 'Gt', self.image_list[idx].split('.')[0]+'.jpg'))
        else:
            # 其余数据集(如 CSD)输入输出文件名完全一致
            label = Image.open(os.path.join(self.image_dir, 'Gt', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)   # 成对变换(同一随机参数)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name   # 测试时附带文件名,便于保存结果
        return image, label


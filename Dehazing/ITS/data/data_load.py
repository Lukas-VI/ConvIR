import os
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, data='ITS', use_transform=True):
    """训练数据加载器(去雾版):按数据集类型对输入/标签做随机裁剪、水平翻转、转张量的数据增强。
    额外接收 data 参数以决定裁剪尺寸与数据目录结构。"""
    image_dir = os.path.join(path, 'train')

    if data == 'real_haze':
        crop_size = [800,1184]   # 真实雾图较大,裁剪尺寸也较大
    else:
        crop_size = 256          # 合成数据裁剪到 256x256

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(crop_size),       # 随机裁剪
                PairRandomHorizontalFilp(),      # 随机水平翻转
                PairToTensor()                   # PIL -> Tensor
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
    """测试数据加载器(去雾版):读 'test' 子目录,is_test 以返回文件名供保存结果。"""
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, data, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, data, batch_size=1, num_workers=0):
    """验证数据加载器(去雾版):同测试集目录 'test',不做变换。"""
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test'), data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    """去雾成对图像数据集:不同数据集(input==hazy, label==gt)目录结构与文件名对应关系不同,按 data 区分。"""
    def __init__(self, image_dir, data, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))  # 列出所有 hazy(雾图)文件名
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.data = data

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 依据数据集类型确定 label(gt/干净图)的取值方式
        if self.data == 'ITS':
            image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx]))            # 雾图
            label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.png'))  # 干净图:取文件名前缀
        elif self.data == 'real_haze':
            image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx]))            # 雾图
            label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx]).replace('hazy', 'GT'))  # 干净图:把路径中 hazy 换成 GT
        elif self.data == 'haze4k':
            image = Image.open(os.path.join(self.image_dir, 'IN', self.image_list[idx]))   # 雾图在 IN 目录
            label = Image.open(os.path.join(self.image_dir, 'GT', self.image_list[idx]))   # 干净图在 GT 目录

        if self.transform:
            image, label = self.transform(image, label)   # 成对变换(同一随机参数)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name   # 测试时附带文件名,便于保存结果
        return image, label


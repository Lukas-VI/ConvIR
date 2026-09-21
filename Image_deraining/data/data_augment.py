import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):
    """成对随机裁剪:输入图和标签图用相同的裁剪参数裁剪,保证空间一一对齐。

    相比 torchvision 的 RandomCrop,额外处理了"图小于裁剪尺寸"的情况(先反射填充补齐)。
    """

    def __call__(self, image, label):

        w,h = image.size[0], image.size[1]
        padw = self.size[0]-w if w<self.size[0] else 0   # 宽不足时需补齐的宽度
        padh = self.size[0]-h if h<self.size[0] else 0   # 高不足时需补齐的高度
        if padw!=0 or padh!=0:
            # 用反射填充把图补到至少与裁剪尺寸一致(右侧/底部补)
            image = F.pad(image, (0,0,padw,padh), padding_mode='reflect')
            label = F.pad(label, (0,0,padw,padh), padding_mode='reflect')

        i, j, h, w = self.get_params(image, self.size)  # 由父类取随机裁剪的左上角与尺寸

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)

class PairCenterCrop(transforms.CenterCrop):
    """成对中心裁剪:输入和标签都从中心裁到 size x size(验证阶段使用,确定性强)。"""

    def __call__(self, image, lable):

        image = F.center_crop(image, (self.size[0], self.size[0]))
        lable = F.center_crop(lable, (self.size[0], self.size[0]))

        return image, lable


class PairCompose(transforms.Compose):
    """成对组合:依次把列表中的变换应用到 (image, label),并保持成对一致。"""

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    """成对随机水平翻转:以概率 self.p 同时翻转输入和标签(随机数只取一次保证同步)。"""

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    """成对转张量:把输入和标签都转为 Tensor(取值范围 0~1)。"""

    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)
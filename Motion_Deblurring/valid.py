import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f


def _valid(model, args, ep):
    """验证函数(去运动模糊版):反射填充保证尺寸可被 32 整除,
    GoPro 验证集上用原图尺度预测(输出索引2)计算 PSNR。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gopro = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)  # 加载验证集
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():      # 验证阶段无需梯度
        print('Start GoPro Evaluation')
        factor = 32            # 网络下采样总步长,输入需是它的倍数
        for idx, data in enumerate(gopro):
            input_img, label_img = data
            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            # 把 H/W 向上取整到 factor(32) 的倍数
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0   # 需补的高空余
            padw = W-w if w%factor!=0 else 0   # 需补的宽空余
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')  # 反射填充到整数倍

            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))     # 按 epoch 建结果目录

            pred = model(input_img)[2]     # 取原图尺度输出
            pred = pred[:,:,:h,:w]         # 裁回原始尺寸(去掉填充)

            pred_clip = torch.clamp(pred, 0, 1)  # 约束到 [0,1]
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)  # skimage 的 PSNR

            psnr_adder(psnr)
            print('\r%03d'%idx, end=' ')   # 进度提示

    print('\n')
    model.train()                          # 恢复训练模式
    return psnr_adder.average()            # 返回平均 PSNR
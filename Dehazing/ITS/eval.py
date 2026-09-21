import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
from pytorch_msssim import ssim
import torch.nn.functional as f

from skimage import img_as_ubyte
import cv2

def _eval(model, args):
    """测试函数(去雾版):加载指定权重,在测试集上计算 PSNR 与 SSIM(pytorch_msssim),
    可选保存输出图像,并统计平均推理时间。"""
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])   # 加载测试权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()   # 清空缓存,释放显存
    adder = Adder()            # 记录单张推理耗时
    model.eval()
    factor = 32                # 网络下采样总步长,输入需是它的倍数
    with torch.no_grad():      # 测试阶段关闭梯度
        psnr_adder = Adder()
        ssim_adder = Adder()

        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            # 把 H/W 向上取整到 factor(32) 的倍数
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0   # 需补的高空余
            padw = W-w if w%factor!=0 else 0   # 需补的宽空余
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')  # 反射填充到整数倍

            tm = time.time()   # 计时开始

            pred = model(input_img)[2]   # 取原图尺度输出
            pred = pred[:,:,:h,:w]       # 裁回原始尺寸(去掉填充)

            elapsed = time.time() - tm   # 单张耗时
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)   # 约束到 [0,1]

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            label_img = (label_img).cuda()
            # PSNR:用 MSE 公式直接算
            psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))
            # SSIM:为控制计算量,把图像自适应池化到约 256 以下再算
            down_ratio = max(1, round(min(H, W) / 256))	
            ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))), 
                            f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False)	
            print('%d iter PSNR_dehazing: %.2f ssim: %f' % (iter_idx + 1, psnr_val, ssim_val))
            ssim_adder(ssim_val)

            if args.save_image:
                # 可选保存输出图片
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255   # 四舍五入误差补偿
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)
            
            # 也用 skimage 的 PSNR 再算一次作为对照
            psnr_mimo = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr_val)

            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr_mimo, elapsed))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))   # 平均 PSNR
        print('The average SSIM is %.5f dB' % (ssim_adder.average()))   # 平均 SSIM

        print("Average time: %f" % adder.average())   # 平均推理时间


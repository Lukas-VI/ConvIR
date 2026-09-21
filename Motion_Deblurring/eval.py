import os
import torch
from torchvision.transforms import functional as F
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
import torch.nn.functional as f

factor = 32   # 网络下采样总步长,输入需是它的倍数

def _eval(model, args):
    """测试函数(去运动模糊版):加载 GoPro 权重,计算 PSNR(skimage)并统计每帧耗时,
    可选保存输出图像。"""
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])   # 加载测试权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)  # 加载测试集
    adder = Adder()
    model.eval()

    with torch.no_grad():      # 测试阶段关闭梯度
        psnr_adder = Adder()
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)
            h, w = input_img.shape[2], input_img.shape[3]
            # 把 H/W 向上取整到 factor(32) 的倍数,再反射填充
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0   # 需补的高空余
            padw = W-w if w%factor!=0 else 0   # 需补的宽空余
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')
            tm = time.time()                   # 记录单帧推理耗时

            pred = model(input_img)[2]         # 取原图尺度输出
            pred = pred[:,:,:h,:w]             # 裁回原始尺寸(去掉填充)
            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)  # 约束到 [0,1]
            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            if args.save_image:
                # 可选保存输出图片
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255   # 四舍五入误差补偿
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)
                
            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)  # skimage 的 PSNR
            psnr_adder(psnr)
            print('%d iter PSNR: %.4f time: %f' % (iter_idx + 1, psnr, elapsed))

        print('==========================================================')
        print('The average PSNR is %.4f dB' % (psnr_adder.average()))     # 平均 PSNR
        print("Average time: %f" % adder.average())                        # 平均单帧耗时
import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f


def _valid(model, args, ep):
    """验证函数:在验证集上计算平均 PSNR,用于挑选最优模型与监控训练进度。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gopro = valid_dataloader(args.valid_data, batch_size=1, num_workers=0)
    model.eval()   # 进入评估模式(关闭 BN dropout 的随机行为)
    psnr_adder = Adder()

    with torch.no_grad():   # 关闭梯度计算,省内存且更快
        print('Start Derain Evaluation')
        factor = 32   # 对齐到 32 的整数倍,保证下采样/上采样不会因尺寸问题出错
        for idx, data in enumerate(gopro):
            input_img, label_img = data
            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            # 把尺寸向上取整到 32 的倍数
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')  # 反射填充至对齐尺寸

            # 生成结果保存目录(以 epoch 命名)
            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = model(input_img)[2]         # 取原图尺度的输出
            pred = pred[:,:,:h,:w]             # 裁回原始尺寸(去掉填充)

            pred_clip = torch.clamp(pred, 0, 1)  # 限制到 [0,1]
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)  # PSNR
            psnr_adder(psnr)
            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()   # 恢复训练模式
    return psnr_adder.average()   # 返回平均 PSNR
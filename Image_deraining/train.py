import os
import torch
from data import train_dataloader
from utils import Adder, Timer
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import torch.nn as nn

from warmup_scheduler import GradualWarmupScheduler


def _train(model, args):
    """
    训练主函数:使用空间域 L1 损失 + 频率域 FFT 损失的多尺度监督,
    配合 CosineAnnealing + 预热(warmup)学习率调度。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()   # 空间/频率域都用 L1 损失

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)          # 每个 epoch 的迭代次数
    warmup_epochs=3
    # 先用 warmup 线性上升到初始 lr,再用余弦退火衰减到 1e-6
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()   # 在开始前先手动 step 一次,让 warmup 生效(热身阶段习惯式调用)
    epoch = 1
    if args.resume:
        # 断点续训:恢复模型、优化器与已训练 epoch
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1   # 从下一 epoch 开始

    writer = SummaryWriter()            # TensorBoard 日志
    epoch_pixel_adder = Adder()         # 记录空间损失(按 epoch)
    epoch_fft_adder = Adder()           # 记录频域损失(按 epoch)
    iter_pixel_adder = Adder()          # 按迭代记录
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')            # 计时(分钟)
    iter_timer = Timer('m')
    best_psnr=-1                        # 记录最好的验证 PSNR

    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data   # (B,C,H,W) 输入与对应干净标签
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)         # 多尺度预测 [1/4, 1/2, 原图]
            # 对标签做同样尺度的下采样,以便与多尺度预测对齐监督
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            # 空间域 L1:三个尺度分别算损失并求和
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)
            loss_content = l1+l2+l3

            # ---- 频域损失:对三个尺度的预测与标签逐尺度做 FFT ----
            # 对预测与标签做二维 FFT,并把 实部/虚部 叠加为最后一维([.., H, W, 2])以计算 L1
            label_fft1 = torch.fft.fft2(label_img4, dim=(-2,-1))
            label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

            pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2,-1))
            pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

            label_fft2 = torch.fft.fft2(label_img2, dim=(-2,-1))
            label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

            pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2,-1))
            pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

            label_fft3 = torch.fft.fft2(label_img, dim=(-2,-1))
            label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

            pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2,-1))
            pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

            # 频域三个尺度的 L1 损失
            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1+f2+f3

            # 总损失 = 空间损失 + 0.1*频域损失(频域作辅助约束)
            loss = loss_content + 0.1 * loss_fft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)  # 梯度裁剪防爆炸
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

            # 周期打印并记录到 TensorBoard
            if (iter_idx + 1) % args.print_freq == 0:
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, scheduler.get_lr()[0], iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                # 重置迭代级累加器与计时器
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
        # 每个 epoch 结束都保存带优化器状态的完整检查点(不断覆盖)
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        # 每 save_freq 个 epoch 额外存档一份(便于回溯)
        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()   # 更新学习率(余弦退火)
        # 周期验证,依据验证 PSNR 保存最优模型
        if epoch_idx % args.valid_freq == 0:
            val_rain = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average DeRain PSNR %.2f dB' % (epoch_idx, val_rain))
            writer.add_scalar('PSNR_DeRain', val_rain, epoch_idx)
            if val_rain >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    # 训练结束保存最终模型
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
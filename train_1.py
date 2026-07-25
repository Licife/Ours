#!/usr/bin/env python
import torch
import torch.nn
import torch.optim as optimi
import numpy as np
from model import *
import config as c
import datasets
import modules.Unet_common as common
import warnings
import os
import time
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from pytorch_msssim import ms_ssim, ssim
import torch.nn.functional as F
import lpips

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()
    return noise

class StarINNLoss(nn.Module):
    def __init__(self, alpha_lpips=0.2, alpha_cssim=0.84):
        """
        :param alpha_lpips: 0.15 意味着 L1 占 85%，LPIPS 占 15%。这是防止偏色的黄金比例。
        """
        super(StarINNLoss, self).__init__()
        self.alpha_lpips = alpha_lpips
        self.alpha_cssim = alpha_cssim

        # 加载 LPIPS (VGG backbone)
        self.lpips_model = lpips.LPIPS(net='vgg').eval()
        for param in self.lpips_model.parameters():
            param.requires_grad = False

    def hybrid_loss(self, pred, target):
        """统一的混合损失：纯 L1 + LPIPS (替代 StarINN 的 L1 + MS-SSIM)"""

        # 1. 纯 L1 损失
        loss_l1 = F.l1_loss(pred, target, reduction='mean')

        # 2. LPIPS 损失
        # 【极其致命的修复】：必须使用 torch.clamp！
        # INN 早期极易产生越界像素，如果不截断直接送入 VGG 会导致特征指数级爆炸 (Loss 飙升至数十亿)
        pred_scaled = torch.clamp(pred * 2.0 - 1.0, min=-1.0, max=1.0)
        target_scaled = torch.clamp(target * 2.0 - 1.0, min=-1.0, max=1.0)
        loss_lpips = self.lpips_model(pred_scaled, target_scaled).mean()

        # 3. 混合并拉回到 Sum 量级 (保持您的梯度缩放逻辑)
        mean_loss = (1 - self.alpha_lpips) * loss_l1 + self.alpha_lpips * loss_lpips
        sum_loss = mean_loss * pred.numel()

        return sum_loss

    def low_freq_loss(self, pred_low, target_low):
        loss_l1 = F.l1_loss(pred_low, target_low, reduction='mean')
        loss_cssim = 1 - ssim(pred_low, target_low, data_range=2.0, size_average=True)

        mean_loss = self.alpha_cssim * loss_cssim + (1 - self.alpha_cssim) * loss_l1
        return mean_loss * pred_low.numel()

def calculate_alm_loss(r, z):
    loss_fn = torch.nn.L1Loss(reduction='sum')
    loss = loss_fn(r, z)
    return loss.to(device)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

def load(name):
    print(f"Loading checkpoint: {name}")
    state_dicts = torch.load(name, weights_only=False)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

    if 'epoch' in state_dicts:
        print(f"Starting from epoch {state_dicts['epoch']}")
        return state_dicts['epoch']
    else:
        print('Starting from epoch 0')
    return 0

def log_to_file(msg):
    """打印并保存日志到文件"""
    print(msg)
    log_path = os.path.join(c.LOG_PATH, 'train_log.txt')
    with open(log_path, 'a+') as f:
        f.write(msg + '\n')

def format_duration(seconds):
    """将秒数格式化为 HH:MM:SS。"""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

#####################
# Model initialize: #
#####################
if __name__ == '__main__':
    if not os.path.exists(c.MODEL_PATH):
        os.makedirs(c.MODEL_PATH)

    log_to_file(
        f"=== Config: Validation every {c.val_freq} epoch(s), Saving Checkpoint every {c.save_freq} epoch(s) ===")

    net = Model()
    net.cuda()
    init_model(net)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    para = get_parameter_number(net)
    print(para)
    params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))

    dwt = common.DWT()
    iwt = common.IWT()

    best_psnr = 0.0

    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

    target_lr = c.lr  # 目标学习率 (您配置文件中的 1e-4)
    start_lr = target_lr / 10  # 初始学习率 (1e-5)

    start_epoch = c.trained_epoch
    # 加载逻辑
    if c.tain_next:
        latest_path = os.path.join(c.MODEL_PATH, 'model_checkpoint_00610.pt')
        if os.path.exists(latest_path):
            start_epoch = load(latest_path)  # 接收返回的 epoch
        else:
            start_epoch = load(c.MODEL_PATH + c.suffix)  # 接收返回的 epoch

    for param_group in optim.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = c.lr  # c.lr 是你配置中的初始学习率

    weight_scheduler = optimi.lr_scheduler.CosineAnnealingLR(optim, T_max=c.epochs, eta_min=1e-7, last_epoch=start_epoch) # 余弦退火

    training_start_time = time.time()
    completed_epoch_times = []

    try:
        tb_log_dir = os.path.join(c.LOG_PATH, 'tensorboard_logs')
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard logging started at {tb_log_dir}")

        for i_epoch in range(start_epoch + 1, c.epochs + 1):
            # i_epoch = i_epoch + c.trained_epoch + 1

            if i_epoch <= c.warmup_epochs:
                # 线性增长公式：lr = start_lr + (target_lr - start_lr) * (current_epoch / warmup_epochs)
                warmup_lr = start_lr + (target_lr - start_lr) * (i_epoch / c.warmup_epochs)

                # 手动更新优化器中的学习率
                for param_group in optim.param_groups:
                    param_group['lr'] = warmup_lr

                print(f"==> [Warmup] Epoch {i_epoch}: LR set to {warmup_lr:.2e}")

            epoch_start_time = time.time()
            epoch_start_datetime = datetime.datetime.now()

            # --- 记录器 ---
            loss_history = []
            r_loss_history = []
            g_loss_history = []
            l_loss_history = []
            a_loss_history = []


            train_psnr_s_history = []
            train_psnr_c_history = []

            #################
            #     train:    #
            #################
            loop = tqdm(enumerate(datasets.trainloader), total=len(datasets.trainloader), leave=True)

            # starinn_loss_fn = StarINNLoss(alpha=0.84).to(device)
            starinn_loss_fn = StarINNLoss(alpha_lpips=0.2).to(device)

            for i_batch, data in loop:
                data = data.to(device)
                cover = data[data.shape[0] // 2:]
                secret = data[:data.shape[0] // 2]
                cover_input = dwt(cover)
                secret_input = dwt(secret)


                input_img = torch.cat((cover_input, secret_input), 1)

                #################
                #    forward:   #
                #################
                # output = net(input_img)
                output, z_pred = net(input_img)
                output_steg = output.narrow(1, 0, 4 * c.channels_in)
                output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                steg_img = iwt(output_steg)

                #################
                #   backward:   #
                #################
                # output_z_guass = gauss_noise(output_z.shape)
                # output_rev = torch.cat((output_steg, output_z_guass), 1)
                # output_image = net(output_rev, rev=True)
                output_image = net(output_steg, rev=True)
                secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                secret_rev = iwt(secret_rev)

                #################
                #     loss:     #
                #################
                g_loss = starinn_loss_fn.hybrid_loss(steg_img, cover)
                # 2. 恢复损失 (Secret vs Secret_rev)
                r_loss = starinn_loss_fn.hybrid_loss(secret_rev, secret)

                # 3. 低频损失 (LL subbands)
                steg_low = output_steg.narrow(1, 0, c.channels_in)
                cover_low = cover_input.narrow(1, 0, c.channels_in)
                l_loss = starinn_loss_fn.low_freq_loss(steg_low, cover_low)

                # 4. ALM / latent consistency loss
                a_loss = calculate_alm_loss(output_z, z_pred)

                # 总损失由恢复、引导、低频和 ALM 损失组成
                total_loss = (c.lamda_reconstruction * r_loss
                              + c.lamda_guide * g_loss
                              + c.lamda_low_frequency * l_loss
                              + c.lamda_alm * a_loss)
                total_loss.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=20)  # 先设个巨大的数，只为了看数值
                # print(f"Current Gradient Norm: {total_norm:.2f}")

                optim.step()
                optim.zero_grad()

                # --- 记录 Loss ---
                loss_history.append(total_loss.item())
                r_loss_history.append(r_loss.item())
                g_loss_history.append(g_loss.item())
                l_loss_history.append(l_loss.item())
                a_loss_history.append(a_loss.item())


                # --- 计算并显示训练集实时 PSNR (不计算梯度) ---
                with torch.no_grad():
                    # 处理 Secret
                    secret_rev_np = secret_rev.detach().cpu().numpy().squeeze() * 255
                    secret_np = secret.detach().cpu().numpy().squeeze() * 255
                    secret_rev_np = np.clip(secret_rev_np, 0, 255)
                    secret_np = np.clip(secret_np, 0, 255)
                    tp_s = computePSNR(secret_rev_np, secret_np)
                    train_psnr_s_history.append(tp_s)

                    # 处理 Cover
                    steg_np = steg_img.detach().cpu().numpy().squeeze() * 255
                    cover_np = cover.detach().cpu().numpy().squeeze() * 255
                    steg_np = np.clip(steg_np, 0, 255)
                    cover_np = np.clip(cover_np, 0, 255)
                    tp_c = computePSNR(cover_np, steg_np)
                    train_psnr_c_history.append(tp_c)

                # 更新进度条
                loop.set_description(f"Train Epoch [{i_epoch}/{c.epochs}]")
                loop.set_postfix(
                    Loss=f"{total_loss.item():.2f}",
                    PSNR_S=f"{tp_s:.2f}",
                    PSNR_C=f"{tp_c:.2f}"
                )

            epoch_train_seconds = time.time() - epoch_start_time
            completed_epoch_times.append(epoch_train_seconds)

            # 计算本 Epoch 平均值
            avg_loss = np.mean(loss_history)
            avg_r_loss = np.mean(r_loss_history)
            avg_g_loss = np.mean(g_loss_history)
            avg_l_loss = np.mean(l_loss_history)
            avg_a_loss = np.mean(a_loss_history)

            avg_train_psnr_s = np.mean(train_psnr_s_history)
            avg_train_psnr_c = np.mean(train_psnr_c_history)

            current_lr = optim.param_groups[0]['lr']

            writer.add_scalar('PSNR_Train/Cover', avg_train_psnr_c, i_epoch)
            writer.add_scalar('PSNR_Train/Secret', avg_train_psnr_s, i_epoch)

            # 同时记录 Loss
            writer.add_scalar('Loss/Total', avg_loss, i_epoch)
            writer.add_scalar('Loss/Reconstruction', avg_r_loss, i_epoch)
            writer.add_scalar('Loss/Guide', avg_g_loss, i_epoch)
            writer.add_scalar('Loss/LowFreq', avg_l_loss, i_epoch)
            writer.add_scalar('Loss/ALM', avg_a_loss, i_epoch)
            writer.add_scalar('Training/Learning_Rate', current_lr, i_epoch)

            #################
            #     val:      #
            #################
            avg_val_psnr_s = 0.0
            avg_val_psnr_c = 0.0

            # 使用 val_interval 变量控制验证频率
            if i_epoch % c.val_freq == 0:
                with torch.no_grad():
                    psnr_s = []
                    psnr_c = []
                    net.eval()

                    val_loop = tqdm(datasets.testloader, desc="Validating", leave=False)

                    for x in val_loop:
                        x = x.to(device)
                        cover = x[x.shape[0] // 2:, :, :, :]
                        secret = x[:x.shape[0] // 2, :, :, :]
                        cover_input = dwt(cover)
                        secret_input = dwt(secret)

                        input_img = torch.cat((cover_input, secret_input), 1)

                        # output = net(input_img)
                        output, _ = net(input_img)
                        output_steg = output.narrow(1, 0, 4 * c.channels_in)
                        steg = iwt(output_steg)
                        # output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                        # output_z = gauss_noise(output_z.shape)

                        output_steg = output_steg.cuda()
                        # output_rev = torch.cat((output_steg, output_z), 1)
                        # output_image = net(output_rev, rev=True)
                        output_image = net(output_steg, rev=True)
                        secret_rev = output_image.narrow(1, 4 * c.channels_in,
                                                         output_image.shape[1] - 4 * c.channels_in)
                        secret_rev = iwt(secret_rev)

                        secret_rev = secret_rev.cpu().numpy().squeeze() * 255
                        np.clip(secret_rev, 0, 255)
                        secret = secret.cpu().numpy().squeeze() * 255
                        np.clip(secret, 0, 255)
                        cover = cover.cpu().numpy().squeeze() * 255
                        np.clip(cover, 0, 255)
                        steg = steg.cpu().numpy().squeeze() * 255
                        np.clip(steg, 0, 255)

                        p_s = computePSNR(secret_rev, secret)
                        p_c = computePSNR(cover, steg)
                        psnr_s.append(p_s)
                        psnr_c.append(p_c)

                        val_loop.set_postfix(Val_S=f"{p_s:.2f}", Val_C=f"{p_c:.2f}")

                    avg_val_psnr_s = np.mean(psnr_s)
                    avg_val_psnr_c = np.mean(psnr_c)

                    # 保存最佳模型
                    if avg_val_psnr_s > best_psnr:
                        best_psnr = avg_val_psnr_s
                        torch.save({'opt': optim.state_dict(),
                                    'net': net.state_dict(),
                                    'epoch': i_epoch,
                                    'best_psnr': best_psnr},
                                   os.path.join(c.MODEL_PATH, 'model_best.pt'))
                        log_to_file(f"    [Info] Saved Best Model (PSNR: {best_psnr:.4f})")

            # 构造与多阶段脚本一致的分行训练日志。
            # 当前文件是单阶段 train_1，因此仅输出 G1/R1/LF1/Z1、C-S1/S1-R1 和 Stage1。
            epoch_end_datetime = datetime.datetime.now()
            total_elapsed_seconds = time.time() - training_start_time

            average_epoch_seconds = float(np.mean(completed_epoch_times)) if completed_epoch_times else epoch_train_seconds
            remaining_epochs = max(c.epochs - i_epoch, 0)
            estimated_finish_datetime = epoch_end_datetime + datetime.timedelta(
                seconds=average_epoch_seconds * remaining_epochs
            )

            log_msg = (
                f"Epoch {i_epoch} Training Done.\n"
                f"  [Loss] Total: {avg_loss:.4f} | "
                f"G1: {avg_g_loss:.4f} | R1: {avg_r_loss:.4f} | "
                f"LF1: {avg_l_loss:.4f} | Z1: {avg_a_loss:.4f}\n"
                f"  [PSNR] C-S1: {avg_train_psnr_c:.2f} dB | "
                f"S1-R1: {avg_train_psnr_s:.2f} dB\n"
                f"  [LR] Stage1: {current_lr:.9f}\n"
                f"  [Time] Start: {epoch_start_datetime.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"End: {epoch_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"  [Elapsed] Epoch Train: {format_duration(epoch_train_seconds)} | "
                f"Total: {format_duration(total_elapsed_seconds)} | "
                f"Estimated Finish: {estimated_finish_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            writer.add_scalar('PSNR_Val/Cover', avg_val_psnr_c, i_epoch)
            writer.add_scalar('PSNR_Val/Secret', avg_val_psnr_s, i_epoch)

            log_to_file(log_msg)

            # --- 保存 Checkpoint (使用 save_interval) ---
            if i_epoch > 0 and (i_epoch % c.save_freq) == 0:
                save_name = c.MODEL_PATH + 'model_checkpoint_%.5i' % i_epoch + '.pt'
                torch.save({'opt': optim.state_dict(),
                            'net': net.state_dict(),
                            'epoch': i_epoch}, save_name)
                print(f"    [Info] Checkpoint saved: {save_name}")

            # --- 保存 Latest (每一轮都存，防止断电) ---
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict(),
                        'epoch': i_epoch},
                       os.path.join(c.MODEL_PATH, 'model_latest.pt'))

            weight_scheduler.step()

        torch.save({'opt': optim.state_dict(),
                    'net': net.state_dict()}, c.MODEL_PATH + 'model_final.pt')

        writer.close()

    except Exception as e:
        log_to_file(f"Error occurred: {e}")
        if c.checkpoint_on_error:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, c.MODEL_PATH + 'model_ABORT.pt')
        raise

    finally:
        pass
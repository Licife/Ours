#!/usr/bin/env python
import datetime
import math
import os
import time
import warnings

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimi
from pytorch_msssim import ssim
from tqdm import tqdm

import config as c
import datasets
import modules.Unet_common as common
from model import Model_1, Model_2, Model_3, init_model

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class StarINNLoss(nn.Module):
    def __init__(self, alpha_lpips=0.2, alpha_cssim=0.84):
        super(StarINNLoss, self).__init__()
        self.alpha_lpips = alpha_lpips
        self.alpha_cssim = alpha_cssim
        self.lpips_model = lpips.LPIPS(net="vgg").eval()
        for param in self.lpips_model.parameters():
            param.requires_grad = False

    def hybrid_losses(self, pairs):
        preds = torch.cat([pred for pred, _ in pairs], dim=0)
        targets = torch.cat([target for _, target in pairs], dim=0)
        preds_scaled = torch.clamp(preds * 2.0 - 1.0, min=-1.0, max=1.0)
        targets_scaled = torch.clamp(targets * 2.0 - 1.0, min=-1.0, max=1.0)
        lpips_values = self.lpips_model(preds_scaled, targets_scaled).reshape(preds.shape[0], -1).mean(dim=1)

        losses = []
        start = 0
        for pred, target in pairs:
            batch_size = pred.shape[0]
            loss_l1 = F.l1_loss(pred, target, reduction="mean")
            loss_lpips = lpips_values[start:start + batch_size].mean()
            losses.append(((1.0 - self.alpha_lpips) * loss_l1 + self.alpha_lpips * loss_lpips) * pred.numel())
            start += batch_size
        return losses

    def low_freq_loss(self, pred_low, target_low):
        loss_l1 = F.l1_loss(pred_low, target_low, reduction="mean")
        loss_cssim = 1.0 - ssim(pred_low, target_low, data_range=2.0, size_average=True)
        return (self.alpha_cssim * loss_cssim + (1.0 - self.alpha_cssim) * loss_l1) * pred_low.numel()


def calculate_alm_loss(output_z, z_pred):
    return F.l1_loss(output_z, z_pred, reduction="sum")


def get_parameter_number(net):
    total_num = sum(param.numel() for param in net.parameters())
    trainable_num = sum(param.numel() for param in net.parameters() if param.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


def computePSNR(origin, pred):
    origin = np.asarray(origin, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    mse = np.mean((origin - pred) ** 2)
    if mse < 1.0e-10:
        return 100.0
    return 10.0 * math.log10((255.0 ** 2) / mse)


def tensor_psnr(origin, pred):
    origin_np = np.clip(origin.detach().cpu().numpy() * 255.0, 0.0, 255.0)
    pred_np = np.clip(pred.detach().cpu().numpy() * 255.0, 0.0, 255.0)
    return computePSNR(origin_np, pred_np)


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log_to_file(message):
    print(message)
    with open(os.path.join(c.LOG_PATH, "train_3_log.txt"), "a+", encoding="utf-8") as file:
        file.write(message + "\n")


def save_model(path, net, optimizer, epoch, best_psnr=None):
    state = {"opt": optimizer.state_dict(), "net": net.state_dict(), "epoch": epoch}
    if best_psnr is not None:
        state["best_psnr"] = best_psnr
    torch.save(state, path)


def load_model(path, net, optimizer):
    print(f"Loading checkpoint: {path}")
    state_dicts = torch.load(path, map_location=device, weights_only=False)
    network_state_dict = {key: value for key, value in state_dicts["net"].items() if "tmp_var" not in key}
    net.load_state_dict(network_state_dict)
    try:
        optimizer.load_state_dict(state_dicts["opt"])
    except Exception:
        print("Cannot load optimizer for some reason or other")
    return int(state_dicts.get("epoch", 0)), float(state_dicts.get("best_psnr", 0.0))


def split_four(data):
    if data.shape[0] % 4 != 0:
        raise ValueError(f"The batch size must be divisible by 4 for cover, secret_1, secret_2 and secret_3, but got {data.shape[0]}.")
    group_size = data.shape[0] // 4
    cover = data[:group_size]
    secret_1 = data[group_size:2 * group_size]
    secret_2 = data[2 * group_size:3 * group_size]
    secret_3 = data[3 * group_size:4 * group_size]
    return cover, secret_1, secret_2, secret_3


if __name__ == "__main__":
    os.makedirs(c.MODEL_PATH_3, exist_ok=True)
    os.makedirs(c.LOG_PATH, exist_ok=True)
    log_to_file(f"=== Config: Validation every {c.val_freq} epoch(s), Saving Checkpoint every {c.save_freq} epoch(s) ===")

    net1 = Model_1().to(device)
    net2 = Model_2().to(device)
    net3 = Model_3().to(device)
    init_model(net1)
    init_model(net2)
    init_model(net3)
    net1 = torch.nn.DataParallel(net1, device_ids=c.device_ids)
    net2 = torch.nn.DataParallel(net2, device_ids=c.device_ids)
    net3 = torch.nn.DataParallel(net3, device_ids=c.device_ids)

    print("Stage 1:", get_parameter_number(net1))
    print("Stage 2:", get_parameter_number(net2))
    print("Stage 3:", get_parameter_number(net3))

    params_trainable1 = list(filter(lambda param: param.requires_grad, net1.parameters()))
    params_trainable2 = list(filter(lambda param: param.requires_grad, net2.parameters()))
    params_trainable3 = list(filter(lambda param: param.requires_grad, net3.parameters()))
    optim1 = torch.optim.Adam(params_trainable1, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    optim2 = torch.optim.Adam(params_trainable2, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    optim3 = torch.optim.Adam(params_trainable3, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

    start_epoch = c.trained_epoch
    best_psnr = 0.0
    if c.tain_next:
        epoch_1, best_psnr_1 = load_model(os.path.join(c.MODEL_PATH_3, "model_latest_1.pt"), net1, optim1)
        epoch_2, best_psnr_2 = load_model(os.path.join(c.MODEL_PATH_3, "model_latest_2.pt"), net2, optim2)
        epoch_3, best_psnr_3 = load_model(os.path.join(c.MODEL_PATH_3, "model_latest_3.pt"), net3, optim3)
        if epoch_1 != epoch_2 or epoch_1 != epoch_3:
            raise RuntimeError(f"The three latest checkpoints have different epochs: {epoch_1}, {epoch_2} and {epoch_3}.")
        start_epoch = epoch_1
        best_psnr = max(best_psnr_1, best_psnr_2, best_psnr_3)

    for optimizer in (optim1, optim2, optim3):
        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = c.lr

    weight_scheduler1 = optimi.lr_scheduler.CosineAnnealingLR(optim1, T_max=c.epochs, eta_min=1e-7, last_epoch=start_epoch)
    weight_scheduler2 = optimi.lr_scheduler.CosineAnnealingLR(optim2, T_max=c.epochs, eta_min=1e-7, last_epoch=start_epoch)
    weight_scheduler3 = optimi.lr_scheduler.CosineAnnealingLR(optim3, T_max=c.epochs, eta_min=1e-7, last_epoch=start_epoch)

    dwt = common.DWT()
    iwt = common.IWT()
    starinn_loss_fn = StarINNLoss(alpha_lpips=0.2).to(device)
    starinn_loss_fn.eval()

    training_start_time = time.time()
    completed_epoch_times = []

    for i_epoch in range(start_epoch + 1, c.epochs + 1):
        epoch_start_time = time.time()
        epoch_start_datetime = datetime.datetime.now()

        loss_history = []
        g1_history, g2_history, g3_history = [], [], []
        r1_history, r2_history, r3_history = [], [], []
        lf1_history, lf2_history, lf3_history = [], [], []
        z1_history, z2_history, z3_history = [], [], []
        train_psnr_c1_history, train_psnr_c2_history, train_psnr_c3_history = [], [], []
        train_psnr_s1_history, train_psnr_s2_history, train_psnr_s3_history = [], [], []

        net1.train()
        net2.train()
        net3.train()
        loop = tqdm(enumerate(datasets.trainloader), total=len(datasets.trainloader), leave=True)

        for _, data in loop:
            data = data.to(device)
            cover, secret_1, secret_2, secret_3 = split_four(data)

            cover_dwt = dwt(cover)
            secret_dwt_1 = dwt(secret_1)
            secret_dwt_2 = dwt(secret_2)
            secret_dwt_3 = dwt(secret_3)

            input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1), dim=1)
            output_dwt_1, z_pred_1 = net1(input_dwt_1)
            output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)
            output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, output_dwt_1.shape[1] - 4 * c.channels_in)
            output_steg_1 = iwt(output_steg_dwt_1)

            input_dwt_2 = torch.cat((output_steg_dwt_1, secret_dwt_2), dim=1)
            output_dwt_2, z_pred_2 = net2(input_dwt_2)
            output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)
            output_z_dwt_2 = output_dwt_2.narrow(1, 4 * c.channels_in, output_dwt_2.shape[1] - 4 * c.channels_in)
            output_steg_2 = iwt(output_steg_dwt_2)

            input_dwt_3 = torch.cat((output_steg_dwt_2, secret_dwt_3), dim=1)
            output_dwt_3, z_pred_3 = net3(input_dwt_3)
            output_steg_dwt_3 = output_dwt_3.narrow(1, 0, 4 * c.channels_in)
            output_z_dwt_3 = output_dwt_3.narrow(1, 4 * c.channels_in, output_dwt_3.shape[1] - 4 * c.channels_in)
            output_steg_3 = iwt(output_steg_dwt_3)

            rev_dwt_3 = net3(output_steg_dwt_3, rev=True)
            rev_steg_dwt_2 = rev_dwt_3.narrow(1, 0, 4 * c.channels_in)
            rev_secret_dwt_3 = rev_dwt_3.narrow(1, 4 * c.channels_in, rev_dwt_3.shape[1] - 4 * c.channels_in)
            rev_secret_3 = iwt(rev_secret_dwt_3)

            rev_dwt_2 = net2(rev_steg_dwt_2, rev=True)
            rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)
            rev_secret_dwt_2 = rev_dwt_2.narrow(1, 4 * c.channels_in, rev_dwt_2.shape[1] - 4 * c.channels_in)
            rev_secret_2 = iwt(rev_secret_dwt_2)

            rev_dwt_1 = net1(rev_steg_dwt_1, rev=True)
            rev_secret_dwt_1 = rev_dwt_1.narrow(1, 4 * c.channels_in, rev_dwt_1.shape[1] - 4 * c.channels_in)
            rev_secret_1 = iwt(rev_secret_dwt_1)

            pairs = [(output_steg_1, cover), (output_steg_2, cover), (output_steg_3, cover),
                     (rev_secret_1, secret_1), (rev_secret_2, secret_2), (rev_secret_3, secret_3)]
            g_loss_1, g_loss_2, g_loss_3, r_loss_1, r_loss_2, r_loss_3 = starinn_loss_fn.hybrid_losses(pairs)

            cover_low = cover_dwt.narrow(1, 0, c.channels_in)
            steg_low_1 = output_steg_dwt_1.narrow(1, 0, c.channels_in)
            steg_low_2 = output_steg_dwt_2.narrow(1, 0, c.channels_in)
            steg_low_3 = output_steg_dwt_3.narrow(1, 0, c.channels_in)
            lf_loss_1 = starinn_loss_fn.low_freq_loss(steg_low_1, cover_low)
            lf_loss_2 = starinn_loss_fn.low_freq_loss(steg_low_2, cover_low)
            lf_loss_3 = starinn_loss_fn.low_freq_loss(steg_low_3, cover_low)
            z_loss_1 = calculate_alm_loss(output_z_dwt_1, z_pred_1)
            z_loss_2 = calculate_alm_loss(output_z_dwt_2, z_pred_2)
            z_loss_3 = calculate_alm_loss(output_z_dwt_3, z_pred_3)

            stage_1_loss = c.lamda_reconstruction * r_loss_1 + c.lamda_guide * g_loss_1 + c.lamda_low_frequency * lf_loss_1 + c.lamda_alm * z_loss_1
            stage_2_loss = c.lamda_reconstruction * r_loss_2 + c.lamda_guide * g_loss_2 + c.lamda_low_frequency * lf_loss_2 + c.lamda_alm * z_loss_2
            stage_3_loss = c.lamda_reconstruction * r_loss_3 + c.lamda_guide * g_loss_3 + c.lamda_low_frequency * lf_loss_3 + c.lamda_alm * z_loss_3
            total_loss = stage_1_loss + stage_2_loss + stage_3_loss

            total_loss.backward()
            optim1.step()
            optim2.step()
            optim3.step()
            optim1.zero_grad(set_to_none=True)
            optim2.zero_grad(set_to_none=True)
            optim3.zero_grad(set_to_none=True)

            loss_history.append(total_loss.item())
            g1_history.append(g_loss_1.item())
            g2_history.append(g_loss_2.item())
            g3_history.append(g_loss_3.item())
            r1_history.append(r_loss_1.item())
            r2_history.append(r_loss_2.item())
            r3_history.append(r_loss_3.item())
            lf1_history.append(lf_loss_1.item())
            lf2_history.append(lf_loss_2.item())
            lf3_history.append(lf_loss_3.item())
            z1_history.append(z_loss_1.item())
            z2_history.append(z_loss_2.item())
            z3_history.append(z_loss_3.item())

            with torch.no_grad():
                psnr_c1 = tensor_psnr(cover, output_steg_1)
                psnr_c2 = tensor_psnr(cover, output_steg_2)
                psnr_c3 = tensor_psnr(cover, output_steg_3)
                psnr_s1 = tensor_psnr(secret_1, rev_secret_1)
                psnr_s2 = tensor_psnr(secret_2, rev_secret_2)
                psnr_s3 = tensor_psnr(secret_3, rev_secret_3)
                train_psnr_c1_history.append(psnr_c1)
                train_psnr_c2_history.append(psnr_c2)
                train_psnr_c3_history.append(psnr_c3)
                train_psnr_s1_history.append(psnr_s1)
                train_psnr_s2_history.append(psnr_s2)
                train_psnr_s3_history.append(psnr_s3)

            loop.set_description(f"Train Epoch [{i_epoch}/{c.epochs}]")
            loop.set_postfix(Loss=f"{total_loss.item():.2f}", C_S3=f"{psnr_c3:.2f}", S1_R1=f"{psnr_s1:.2f}", S2_R2=f"{psnr_s2:.2f}", S3_R3=f"{psnr_s3:.2f}")

        epoch_train_seconds = time.time() - epoch_start_time
        completed_epoch_times.append(epoch_train_seconds)
        avg_loss = float(np.mean(loss_history))
        avg_g1, avg_g2, avg_g3 = float(np.mean(g1_history)), float(np.mean(g2_history)), float(np.mean(g3_history))
        avg_r1, avg_r2, avg_r3 = float(np.mean(r1_history)), float(np.mean(r2_history)), float(np.mean(r3_history))
        avg_lf1, avg_lf2, avg_lf3 = float(np.mean(lf1_history)), float(np.mean(lf2_history)), float(np.mean(lf3_history))
        avg_z1, avg_z2, avg_z3 = float(np.mean(z1_history)), float(np.mean(z2_history)), float(np.mean(z3_history))
        avg_train_c1 = float(np.mean(train_psnr_c1_history))
        avg_train_c2 = float(np.mean(train_psnr_c2_history))
        avg_train_c3 = float(np.mean(train_psnr_c3_history))
        avg_train_s1 = float(np.mean(train_psnr_s1_history))
        avg_train_s2 = float(np.mean(train_psnr_s2_history))
        avg_train_s3 = float(np.mean(train_psnr_s3_history))

        if i_epoch % c.val_freq == 0:
            net1.eval()
            net2.eval()
            net3.eval()
            psnr_c1_list, psnr_c2_list, psnr_c3_list = [], [], []
            psnr_s1_list, psnr_s2_list, psnr_s3_list = [], [], []

            with torch.no_grad():
                val_loop = tqdm(datasets.testloader, desc="Validating", leave=False)
                for data in val_loop:
                    data = data.to(device)
                    cover, secret_1, secret_2, secret_3 = split_four(data)

                    cover_dwt = dwt(cover)
                    secret_dwt_1 = dwt(secret_1)
                    secret_dwt_2 = dwt(secret_2)
                    secret_dwt_3 = dwt(secret_3)

                    output_dwt_1, _ = net1(torch.cat((cover_dwt, secret_dwt_1), dim=1))
                    output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)
                    output_steg_1 = iwt(output_steg_dwt_1)

                    output_dwt_2, _ = net2(torch.cat((output_steg_dwt_1, secret_dwt_2), dim=1))
                    output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)
                    output_steg_2 = iwt(output_steg_dwt_2)

                    output_dwt_3, _ = net3(torch.cat((output_steg_dwt_2, secret_dwt_3), dim=1))
                    output_steg_dwt_3 = output_dwt_3.narrow(1, 0, 4 * c.channels_in)
                    output_steg_3 = iwt(output_steg_dwt_3)

                    rev_dwt_3 = net3(output_steg_dwt_3, rev=True)
                    rev_steg_dwt_2 = rev_dwt_3.narrow(1, 0, 4 * c.channels_in)
                    rev_secret_3 = iwt(rev_dwt_3.narrow(1, 4 * c.channels_in, rev_dwt_3.shape[1] - 4 * c.channels_in))

                    rev_dwt_2 = net2(rev_steg_dwt_2, rev=True)
                    rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)
                    rev_secret_2 = iwt(rev_dwt_2.narrow(1, 4 * c.channels_in, rev_dwt_2.shape[1] - 4 * c.channels_in))

                    rev_dwt_1 = net1(rev_steg_dwt_1, rev=True)
                    rev_secret_1 = iwt(rev_dwt_1.narrow(1, 4 * c.channels_in, rev_dwt_1.shape[1] - 4 * c.channels_in))

                    p_c1 = tensor_psnr(cover, output_steg_1)
                    p_c2 = tensor_psnr(cover, output_steg_2)
                    p_c3 = tensor_psnr(cover, output_steg_3)
                    p_s1 = tensor_psnr(secret_1, rev_secret_1)
                    p_s2 = tensor_psnr(secret_2, rev_secret_2)
                    p_s3 = tensor_psnr(secret_3, rev_secret_3)
                    psnr_c1_list.append(p_c1)
                    psnr_c2_list.append(p_c2)
                    psnr_c3_list.append(p_c3)
                    psnr_s1_list.append(p_s1)
                    psnr_s2_list.append(p_s2)
                    psnr_s3_list.append(p_s3)
                    val_loop.set_postfix(C_S3=f"{p_c3:.2f}", S1_R1=f"{p_s1:.2f}", S2_R2=f"{p_s2:.2f}", S3_R3=f"{p_s3:.2f}")

            avg_val_c1 = float(np.mean(psnr_c1_list))
            avg_val_c2 = float(np.mean(psnr_c2_list))
            avg_val_c3 = float(np.mean(psnr_c3_list))
            avg_val_s1 = float(np.mean(psnr_s1_list))
            avg_val_s2 = float(np.mean(psnr_s2_list))
            avg_val_s3 = float(np.mean(psnr_s3_list))
            current_val_psnr = (avg_val_s1 + avg_val_s2 + avg_val_s3) / 3.0

            if current_val_psnr > best_psnr:
                best_psnr = current_val_psnr
                save_model(os.path.join(c.MODEL_PATH_3, "model_best_1.pt"), net1, optim1, i_epoch, best_psnr)
                save_model(os.path.join(c.MODEL_PATH_3, "model_best_2.pt"), net2, optim2, i_epoch, best_psnr)
                save_model(os.path.join(c.MODEL_PATH_3, "model_best_3.pt"), net3, optim3, i_epoch, best_psnr)
                log_to_file(f"    [Info] Saved Best Models (C-S1: {avg_val_c1:.2f} dB | C-S2: {avg_val_c2:.2f} dB | C-S3: {avg_val_c3:.2f} dB | S1-R1: {avg_val_s1:.2f} dB | S2-R2: {avg_val_s2:.2f} dB | S3-R3: {avg_val_s3:.2f} dB)")

        epoch_end_datetime = datetime.datetime.now()
        total_elapsed_seconds = time.time() - training_start_time
        average_epoch_seconds = float(np.mean(completed_epoch_times))
        remaining_epochs = max(c.epochs - i_epoch, 0)
        estimated_finish_datetime = epoch_end_datetime + datetime.timedelta(seconds=average_epoch_seconds * remaining_epochs)

        log_message = (
            f"Epoch {i_epoch} Training Done.\n"
            f"  [Loss] Total: {avg_loss:.4f} | G1: {avg_g1:.4f} | R1: {avg_r1:.4f} | LF1: {avg_lf1:.4f} | Z1: {avg_z1:.4f}\n"
            f"         G2: {avg_g2:.4f} | R2: {avg_r2:.4f} | LF2: {avg_lf2:.4f} | Z2: {avg_z2:.4f}\n"
            f"         G3: {avg_g3:.4f} | R3: {avg_r3:.4f} | LF3: {avg_lf3:.4f} | Z3: {avg_z3:.4f}\n"
            f"  [PSNR] C-S1: {avg_train_c1:.2f} dB | C-S2: {avg_train_c2:.2f} dB | C-S3: {avg_train_c3:.2f} dB | S1-R1: {avg_train_s1:.2f} dB | S2-R2: {avg_train_s2:.2f} dB | S3-R3: {avg_train_s3:.2f} dB\n"
            f"  [LR] Stage1: {optim1.param_groups[0]['lr']:.9f} | Stage2: {optim2.param_groups[0]['lr']:.9f} | Stage3: {optim3.param_groups[0]['lr']:.9f}\n"
            f"  [Time] Start: {epoch_start_datetime.strftime('%Y-%m-%d %H:%M:%S')} | End: {epoch_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  [Elapsed] Epoch Train: {format_duration(epoch_train_seconds)} | Total: {format_duration(total_elapsed_seconds)} | Estimated Finish: {estimated_finish_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        log_to_file(log_message)

        if i_epoch > 0 and i_epoch % c.save_freq == 0:
            save_model(os.path.join(c.MODEL_PATH_3, f"model_checkpoint_{i_epoch:05d}_1.pt"), net1, optim1, i_epoch)
            save_model(os.path.join(c.MODEL_PATH_3, f"model_checkpoint_{i_epoch:05d}_2.pt"), net2, optim2, i_epoch)
            save_model(os.path.join(c.MODEL_PATH_3, f"model_checkpoint_{i_epoch:05d}_3.pt"), net3, optim3, i_epoch)

        save_model(os.path.join(c.MODEL_PATH_3, "model_latest_1.pt"), net1, optim1, i_epoch, best_psnr)
        save_model(os.path.join(c.MODEL_PATH_3, "model_latest_2.pt"), net2, optim2, i_epoch, best_psnr)
        save_model(os.path.join(c.MODEL_PATH_3, "model_latest_3.pt"), net3, optim3, i_epoch, best_psnr)

        weight_scheduler1.step()
        weight_scheduler2.step()
        weight_scheduler3.step()

    save_model(os.path.join(c.MODEL_PATH_3, "model_final_1.pt"), net1, optim1, c.epochs, best_psnr)
    save_model(os.path.join(c.MODEL_PATH_3, "model_final_2.pt"), net2, optim2, c.epochs, best_psnr)
    save_model(os.path.join(c.MODEL_PATH_3, "model_final_3.pt"), net3, optim3, c.epochs, best_psnr)

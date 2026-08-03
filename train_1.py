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
from model import Model_1, init_model

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

    def guide_reconstruction_loss(self, steg_img, cover, secret_rev, secret):
        guide_l1 = F.l1_loss(steg_img, cover, reduction="mean")
        reconstruction_l1 = F.l1_loss(secret_rev, secret, reduction="mean")

        pred = torch.cat((steg_img, secret_rev), dim=0)
        target = torch.cat((cover, secret), dim=0)
        pred_scaled = torch.clamp(pred * 2.0 - 1.0, min=-1.0, max=1.0)
        target_scaled = torch.clamp(target * 2.0 - 1.0, min=-1.0, max=1.0)

        lpips_value = self.lpips_model(pred_scaled, target_scaled).reshape(pred.shape[0], -1).mean(dim=1)
        batch_size = steg_img.shape[0]
        guide_lpips = lpips_value[:batch_size].mean()
        reconstruction_lpips = lpips_value[batch_size:].mean()

        guide_mean = (1.0 - self.alpha_lpips) * guide_l1 + self.alpha_lpips * guide_lpips
        reconstruction_mean = (1.0 - self.alpha_lpips) * reconstruction_l1 + self.alpha_lpips * reconstruction_lpips

        guide_loss = guide_mean * steg_img.numel()
        reconstruction_loss = reconstruction_mean * secret_rev.numel()
        return guide_loss, reconstruction_loss

    def low_freq_loss(self, pred_low, target_low):
        loss_l1 = F.l1_loss(pred_low, target_low, reduction="mean")
        loss_cssim = 1.0 - ssim(pred_low, target_low, data_range=2.0, size_average=True)
        mean_loss = self.alpha_cssim * loss_cssim + (1.0 - self.alpha_cssim) * loss_l1
        return mean_loss * pred_low.numel()


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


def split_two(data):
    pair_count = data.shape[0] // 2
    if pair_count == 0:
        raise ValueError(f"The batch contains too few images: {data.shape[0]}.")
    secret = data[:pair_count]
    cover = data[pair_count:2 * pair_count]
    return cover, secret


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log_to_file(message):
    print(message)
    os.makedirs(c.LOG_PATH, exist_ok=True)
    with open(os.path.join(c.LOG_PATH, "train_1_log.txt"), "a+", encoding="utf-8") as file:
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
    epoch = int(state_dicts.get("epoch", 0))
    best_psnr = float(state_dicts.get("best_psnr", 0.0))
    return epoch, best_psnr


if __name__ == "__main__":
    os.makedirs(c.MODEL_PATH_1, exist_ok=True)
    os.makedirs(c.LOG_PATH, exist_ok=True)

    if c.batch_size % 2 != 0 or c.batchsize_val % 2 != 0:
        print("[Warning] train_1 uses image pairs. The final unpaired image of each odd-sized batch will be ignored.")

    log_to_file(f"=== Config: Validation every {c.val_freq} epoch(s), Saving Checkpoint every {c.save_freq} epoch(s) ===")

    net = Model_1().to(device)
    init_model(net)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    print("Stage 1:", get_parameter_number(net))

    params_trainable = list(filter(lambda param: param.requires_grad, net.parameters()))
    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

    start_epoch = c.trained_epoch
    best_psnr = 0.0
    if c.tain_next:
        latest_path = os.path.join(c.MODEL_PATH_1, "model_latest.pt")
        load_path = latest_path if os.path.exists(latest_path) else c.MODEL_PATH_1 + c.suffix
        start_epoch, best_psnr = load_model(load_path, net, optim)

    for param_group in optim.param_groups:
        if "initial_lr" not in param_group:
            param_group["initial_lr"] = c.lr

    weight_scheduler = optimi.lr_scheduler.CosineAnnealingLR(optim, T_max=c.epochs, eta_min=1e-7, last_epoch=start_epoch)
    dwt = common.DWT()
    iwt = common.IWT()

    starinn_loss_fn = StarINNLoss(alpha_lpips=0.2).to(device)
    starinn_loss_fn.eval()

    training_start_time = time.time()
    completed_epoch_times = []

    try:
        for i_epoch in range(start_epoch + 1, c.epochs + 1):
            epoch_start_time = time.time()
            epoch_start_datetime = datetime.datetime.now()

            loss_history = []
            g1_history = []
            r1_history = []
            lf1_history = []
            z1_history = []
            train_psnr_c1_history = []
            train_psnr_s1_history = []

            net.train()
            loop = tqdm(enumerate(datasets.trainloader), total=len(datasets.trainloader), leave=True)

            for _, data in loop:
                data = data.to(device)
                cover, secret = split_two(data)

                cover_dwt = dwt(cover)
                secret_dwt = dwt(secret)
                input_dwt = torch.cat((cover_dwt, secret_dwt), dim=1)

                output_dwt, z_pred = net(input_dwt)
                output_steg_dwt = output_dwt.narrow(1, 0, 4 * c.channels_in)
                output_z_dwt = output_dwt.narrow(1, 4 * c.channels_in, output_dwt.shape[1] - 4 * c.channels_in)
                output_steg = iwt(output_steg_dwt)

                rev_dwt = net(output_steg_dwt, rev=True)
                rev_secret_dwt = rev_dwt.narrow(1, 4 * c.channels_in, rev_dwt.shape[1] - 4 * c.channels_in)
                rev_secret = iwt(rev_secret_dwt)

                g_loss_1, r_loss_1 = starinn_loss_fn.guide_reconstruction_loss(output_steg, cover, rev_secret, secret)
                cover_low = cover_dwt.narrow(1, 0, c.channels_in)
                steg_low_1 = output_steg_dwt.narrow(1, 0, c.channels_in)
                lf_loss_1 = starinn_loss_fn.low_freq_loss(steg_low_1, cover_low)
                z_loss_1 = calculate_alm_loss(output_z_dwt, z_pred)

                total_loss = c.lamda_reconstruction * r_loss_1 + c.lamda_guide * g_loss_1 + c.lamda_low_frequency * lf_loss_1 + c.lamda_alm * z_loss_1

                optim.zero_grad(set_to_none=True)
                total_loss.backward()
                optim.step()

                loss_history.append(total_loss.item())
                g1_history.append(g_loss_1.item())
                r1_history.append(r_loss_1.item())
                lf1_history.append(lf_loss_1.item())
                z1_history.append(z_loss_1.item())

                with torch.no_grad():
                    psnr_c1 = tensor_psnr(cover, output_steg)
                    psnr_s1 = tensor_psnr(secret, rev_secret)
                    train_psnr_c1_history.append(psnr_c1)
                    train_psnr_s1_history.append(psnr_s1)

                loop.set_description(f"Train Epoch [{i_epoch}/{c.epochs}]")
                loop.set_postfix(Loss=f"{total_loss.item():.2f}", C_S1=f"{psnr_c1:.2f}", S1_R1=f"{psnr_s1:.2f}")

            epoch_train_seconds = time.time() - epoch_start_time
            completed_epoch_times.append(epoch_train_seconds)

            avg_loss = float(np.mean(loss_history))
            avg_g1 = float(np.mean(g1_history))
            avg_r1 = float(np.mean(r1_history))
            avg_lf1 = float(np.mean(lf1_history))
            avg_z1 = float(np.mean(z1_history))
            avg_train_c1 = float(np.mean(train_psnr_c1_history))
            avg_train_s1 = float(np.mean(train_psnr_s1_history))

            if i_epoch % c.val_freq == 0:
                net.eval()
                psnr_c1_list = []
                psnr_s1_list = []

                with torch.no_grad():
                    val_loop = tqdm(datasets.testloader, desc="Validating", leave=False)
                    for data in val_loop:
                        data = data.to(device)
                        cover, secret = split_two(data)

                        cover_dwt = dwt(cover)
                        secret_dwt = dwt(secret)
                        input_dwt = torch.cat((cover_dwt, secret_dwt), dim=1)

                        output_dwt, _ = net(input_dwt)
                        output_steg_dwt = output_dwt.narrow(1, 0, 4 * c.channels_in)
                        output_steg = iwt(output_steg_dwt)

                        rev_dwt = net(output_steg_dwt, rev=True)
                        rev_secret_dwt = rev_dwt.narrow(1, 4 * c.channels_in, rev_dwt.shape[1] - 4 * c.channels_in)
                        rev_secret = iwt(rev_secret_dwt)

                        p_c1 = tensor_psnr(cover, output_steg)
                        p_s1 = tensor_psnr(secret, rev_secret)
                        psnr_c1_list.append(p_c1)
                        psnr_s1_list.append(p_s1)
                        val_loop.set_postfix(C_S1=f"{p_c1:.2f}", S1_R1=f"{p_s1:.2f}")

                avg_val_c1 = float(np.mean(psnr_c1_list))
                avg_val_s1 = float(np.mean(psnr_s1_list))
                log_to_file(f"  [Validation] C-S1: {avg_val_c1:.2f} dB | S1-R1: {avg_val_s1:.2f} dB")

                if avg_val_s1 > best_psnr:
                    best_psnr = avg_val_s1
                    save_model(os.path.join(c.MODEL_PATH_1, "model_best.pt"), net, optim, i_epoch, best_psnr)
                    log_to_file(f"    [Info] Saved Best Model (C-S1: {avg_val_c1:.2f} dB | S1-R1: {avg_val_s1:.2f} dB)")

            epoch_end_datetime = datetime.datetime.now()
            total_elapsed_seconds = time.time() - training_start_time
            average_epoch_seconds = float(np.mean(completed_epoch_times))
            remaining_epochs = max(c.epochs - i_epoch, 0)
            estimated_finish_datetime = epoch_end_datetime + datetime.timedelta(seconds=average_epoch_seconds * remaining_epochs)

            log_message = (
                f"Epoch {i_epoch} Training Done.\n"
                f"  [Loss] Total: {avg_loss:.4f} | G1: {avg_g1:.4f} | R1: {avg_r1:.4f} | LF1: {avg_lf1:.4f} | Z1: {avg_z1:.4f}\n"
                f"  [PSNR] C-S1: {avg_train_c1:.2f} dB | S1-R1: {avg_train_s1:.2f} dB\n"
                f"  [LR] Stage1: {optim.param_groups[0]['lr']:.9f}\n"
                f"  [Time] Start: {epoch_start_datetime.strftime('%Y-%m-%d %H:%M:%S')} | End: {epoch_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"  [Elapsed] Epoch Train: {format_duration(epoch_train_seconds)} | Total: {format_duration(total_elapsed_seconds)} | Estimated Finish: {estimated_finish_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            log_to_file(log_message)

            if i_epoch > 0 and i_epoch % c.save_freq == 0:
                checkpoint_path = os.path.join(c.MODEL_PATH_1, f"model_checkpoint_{i_epoch:05d}.pt")
                save_model(checkpoint_path, net, optim, i_epoch)
                print(f"    [Info] Checkpoint saved: {checkpoint_path}")

            save_model(os.path.join(c.MODEL_PATH_1, "model_latest.pt"), net, optim, i_epoch, best_psnr)
            weight_scheduler.step()

        save_model(os.path.join(c.MODEL_PATH_1, "model_final.pt"), net, optim, c.epochs, best_psnr)

    except Exception as error:
        log_to_file(f"Error occurred: {error}")
        if c.checkpoint_on_error:
            abort_epoch = i_epoch if "i_epoch" in locals() else start_epoch
            save_model(os.path.join(c.MODEL_PATH_1, "model_ABORT.pt"), net, optim, abort_epoch, best_psnr)
        raise

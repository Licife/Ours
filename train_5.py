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
from model import Model_1, Model_2, Model_3, Model_4, Model_5, init_model

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


def split_six(data):
    if data.shape[0] % 6 != 0:
        raise ValueError(f"The batch size must be divisible by 6 for cover and five secret images, but got {data.shape[0]}.")
    group_size = data.shape[0] // 6
    cover = data[:group_size]
    secrets = [data[i * group_size:(i + 1) * group_size] for i in range(1, 6)]
    return cover, secrets


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log_to_file(message):
    print(message)
    with open(os.path.join(c.LOG_PATH, "train_5_log.txt"), "a+", encoding="utf-8") as file:
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


def forward_cascade(nets, cover_dwt, secret_dwts, iwt):
    carrier_dwt = cover_dwt
    steg_dwts, steg_images, output_zs, z_preds = [], [], [], []

    for net, secret_dwt in zip(nets, secret_dwts):
        output_dwt, z_pred = net(torch.cat((carrier_dwt, secret_dwt), dim=1))
        steg_dwt = output_dwt.narrow(1, 0, 4 * c.channels_in)
        output_z = output_dwt.narrow(1, 4 * c.channels_in, output_dwt.shape[1] - 4 * c.channels_in)
        steg_dwts.append(steg_dwt)
        steg_images.append(iwt(steg_dwt))
        output_zs.append(output_z)
        z_preds.append(z_pred)
        carrier_dwt = steg_dwt

    return steg_dwts, steg_images, output_zs, z_preds


def reverse_cascade(nets, final_steg_dwt, iwt):
    recovered_secrets = [None] * len(nets)
    recovered_carrier_dwt = final_steg_dwt

    for index in range(len(nets) - 1, -1, -1):
        reverse_dwt = nets[index](recovered_carrier_dwt, rev=True)
        recovered_carrier_dwt = reverse_dwt.narrow(1, 0, 4 * c.channels_in)
        recovered_secret_dwt = reverse_dwt.narrow(1, 4 * c.channels_in, reverse_dwt.shape[1] - 4 * c.channels_in)
        recovered_secrets[index] = iwt(recovered_secret_dwt)

    return recovered_secrets


if __name__ == "__main__":
    os.makedirs(c.MODEL_PATH_5, exist_ok=True)
    os.makedirs(c.LOG_PATH, exist_ok=True)
    log_to_file(f"=== Config: Validation every {c.val_freq} epoch(s), Saving Checkpoint every {c.save_freq} epoch(s) ===")

    nets = [Model_1().to(device), Model_2().to(device), Model_3().to(device), Model_4().to(device), Model_5().to(device)]
    for net in nets:
        init_model(net)
    nets = [torch.nn.DataParallel(net, device_ids=c.device_ids) for net in nets]

    for index, net in enumerate(nets, start=1):
        print(f"Stage {index}:", get_parameter_number(net))

    optimizers = [torch.optim.Adam(list(filter(lambda param: param.requires_grad, net.parameters())), lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay) for net in nets]

    start_epoch = c.trained_epoch
    best_psnr = 0.0

    if c.tain_next:
        loaded_epochs, loaded_best = [], []
        for index, (net, optimizer) in enumerate(zip(nets, optimizers), start=1):
            epoch, stage_best = load_model(os.path.join(c.MODEL_PATH_5, f"model_latest_{index}.pt"), net, optimizer)
            loaded_epochs.append(epoch)
            loaded_best.append(stage_best)
        if len(set(loaded_epochs)) != 1:
            raise RuntimeError(f"The five latest checkpoints have different epochs: {loaded_epochs}.")
        start_epoch = loaded_epochs[0]
        best_psnr = max(loaded_best)

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = c.lr

    schedulers = [optimi.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs, eta_min=1e-7, last_epoch=start_epoch) for optimizer in optimizers]
    dwt = common.DWT()
    iwt = common.IWT()
    loss_fn = StarINNLoss(alpha_lpips=0.2).to(device)
    loss_fn.eval()

    training_start_time = time.time()
    completed_epoch_times = []

    for i_epoch in range(start_epoch + 1, c.epochs + 1):
        epoch_start_time = time.time()
        epoch_start_datetime = datetime.datetime.now()
        loss_history = []
        g_history = [[] for _ in range(5)]
        r_history = [[] for _ in range(5)]
        lf_history = [[] for _ in range(5)]
        z_history = [[] for _ in range(5)]
        train_cover_psnr = [[] for _ in range(5)]
        train_secret_psnr = [[] for _ in range(5)]

        for net in nets:
            net.train()

        loop = tqdm(enumerate(datasets.trainloader), total=len(datasets.trainloader), leave=True)

        for _, data in loop:
            data = data.to(device)
            cover, secrets = split_six(data)
            cover_dwt = dwt(cover)
            secret_dwts = [dwt(secret) for secret in secrets]

            steg_dwts, steg_images, output_zs, z_preds = forward_cascade(nets, cover_dwt, secret_dwts, iwt)
            recovered_secrets = reverse_cascade(nets, steg_dwts[-1], iwt)

            pairs = [(steg_image, cover) for steg_image in steg_images] + list(zip(recovered_secrets, secrets))
            hybrid_losses = loss_fn.hybrid_losses(pairs)
            g_losses = hybrid_losses[:5]
            r_losses = hybrid_losses[5:]

            cover_low = cover_dwt.narrow(1, 0, c.channels_in)
            lf_losses = [loss_fn.low_freq_loss(steg_dwt.narrow(1, 0, c.channels_in), cover_low) for steg_dwt in steg_dwts]
            z_losses = [calculate_alm_loss(output_z, z_pred) for output_z, z_pred in zip(output_zs, z_preds)]
            stage_losses = [c.lamda_reconstruction * r_losses[i] + c.lamda_guide * g_losses[i] + c.lamda_low_frequency * lf_losses[i] + c.lamda_alm * z_losses[i] for i in range(5)]
            total_loss = sum(stage_losses)

            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            loss_history.append(total_loss.item())
            for index in range(5):
                g_history[index].append(g_losses[index].item())
                r_history[index].append(r_losses[index].item())
                lf_history[index].append(lf_losses[index].item())
                z_history[index].append(z_losses[index].item())

            with torch.no_grad():
                cover_psnr_values = [tensor_psnr(cover, steg_image) for steg_image in steg_images]
                secret_psnr_values = [tensor_psnr(secret, recovered) for secret, recovered in zip(secrets, recovered_secrets)]
                for index in range(5):
                    train_cover_psnr[index].append(cover_psnr_values[index])
                    train_secret_psnr[index].append(secret_psnr_values[index])

            loop.set_description(f"Train Epoch [{i_epoch}/{c.epochs}]")
            loop.set_postfix(Loss=f"{total_loss.item():.2f}", C_S5=f"{cover_psnr_values[4]:.2f}", S1_R1=f"{secret_psnr_values[0]:.2f}", S2_R2=f"{secret_psnr_values[1]:.2f}", S3_R3=f"{secret_psnr_values[2]:.2f}", S4_R4=f"{secret_psnr_values[3]:.2f}", S5_R5=f"{secret_psnr_values[4]:.2f}")

        epoch_train_seconds = time.time() - epoch_start_time
        completed_epoch_times.append(epoch_train_seconds)
        avg_loss = float(np.mean(loss_history))
        avg_g = [float(np.mean(history)) for history in g_history]
        avg_r = [float(np.mean(history)) for history in r_history]
        avg_lf = [float(np.mean(history)) for history in lf_history]
        avg_z = [float(np.mean(history)) for history in z_history]
        avg_train_cover = [float(np.mean(history)) for history in train_cover_psnr]
        avg_train_secret = [float(np.mean(history)) for history in train_secret_psnr]

        if i_epoch % c.val_freq == 0:
            for net in nets:
                net.eval()

            val_cover_psnr = [[] for _ in range(5)]
            val_secret_psnr = [[] for _ in range(5)]

            with torch.no_grad():
                val_loop = tqdm(datasets.testloader, desc="Validating", leave=False)

                for data in val_loop:
                    data = data.to(device)
                    cover, secrets = split_six(data)
                    cover_dwt = dwt(cover)
                    secret_dwts = [dwt(secret) for secret in secrets]

                    steg_dwts, steg_images, _, _ = forward_cascade(nets, cover_dwt, secret_dwts, iwt)
                    recovered_secrets = reverse_cascade(nets, steg_dwts[-1], iwt)
                    cover_psnr_values = [tensor_psnr(cover, steg_image) for steg_image in steg_images]
                    secret_psnr_values = [tensor_psnr(secret, recovered) for secret, recovered in zip(secrets, recovered_secrets)]

                    for index in range(5):
                        val_cover_psnr[index].append(cover_psnr_values[index])
                        val_secret_psnr[index].append(secret_psnr_values[index])

                    val_loop.set_postfix(C_S5=f"{cover_psnr_values[4]:.2f}", S1_R1=f"{secret_psnr_values[0]:.2f}", S2_R2=f"{secret_psnr_values[1]:.2f}", S3_R3=f"{secret_psnr_values[2]:.2f}", S4_R4=f"{secret_psnr_values[3]:.2f}", S5_R5=f"{secret_psnr_values[4]:.2f}")

            avg_val_cover = [float(np.mean(history)) for history in val_cover_psnr]
            avg_val_secret = [float(np.mean(history)) for history in val_secret_psnr]
            current_val_psnr = float(np.mean(avg_val_secret))
            cover_text = " | ".join([f"C-S{i + 1}: {avg_val_cover[i]:.2f} dB" for i in range(5)])
            secret_text = " | ".join([f"S{i + 1}-R{i + 1}: {avg_val_secret[i]:.2f} dB" for i in range(5)])
            log_to_file(f"  [Validation] {cover_text} | {secret_text}")

            if current_val_psnr > best_psnr:
                best_psnr = current_val_psnr
                for index, (net, optimizer) in enumerate(zip(nets, optimizers), start=1):
                    save_model(os.path.join(c.MODEL_PATH_5, f"model_best_{index}.pt"), net, optimizer, i_epoch, best_psnr)
                log_to_file("    [Info] Saved Best Models.")

        epoch_end_datetime = datetime.datetime.now()
        total_elapsed_seconds = time.time() - training_start_time
        average_epoch_seconds = float(np.mean(completed_epoch_times))
        remaining_epochs = max(c.epochs - i_epoch, 0)
        estimated_finish_datetime = epoch_end_datetime + datetime.timedelta(seconds=average_epoch_seconds * remaining_epochs)

        loss_lines = [f"         G{i + 1}: {avg_g[i]:.4f} | R{i + 1}: {avg_r[i]:.4f} | LF{i + 1}: {avg_lf[i]:.4f} | Z{i + 1}: {avg_z[i]:.4f}" for i in range(5)]
        loss_lines[0] = f"  [Loss] Total: {avg_loss:.4f} | G1: {avg_g[0]:.4f} | R1: {avg_r[0]:.4f} | LF1: {avg_lf[0]:.4f} | Z1: {avg_z[0]:.4f}"
        train_cover_text = " | ".join([f"C-S{i + 1}: {avg_train_cover[i]:.2f} dB" for i in range(5)])
        train_secret_text = " | ".join([f"S{i + 1}-R{i + 1}: {avg_train_secret[i]:.2f} dB" for i in range(5)])
        lr_text = " | ".join([f"Stage{i + 1}: {optimizers[i].param_groups[0]['lr']:.9f}" for i in range(5)])

        log_message = (
            f"Epoch {i_epoch} Training Done.\n"
            + "\n".join(loss_lines)
            + f"\n  [PSNR] {train_cover_text} | {train_secret_text}\n"
            + f"  [LR] {lr_text}\n"
            + f"  [Time] Start: {epoch_start_datetime.strftime('%Y-%m-%d %H:%M:%S')} | End: {epoch_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
            + f"  [Elapsed] Epoch Train: {format_duration(epoch_train_seconds)} | Total: {format_duration(total_elapsed_seconds)} | Estimated Finish: {estimated_finish_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        log_to_file(log_message)

        if i_epoch > 0 and i_epoch % c.save_freq == 0:
            for index, (net, optimizer) in enumerate(zip(nets, optimizers), start=1):
                save_model(os.path.join(c.MODEL_PATH_5, f"model_checkpoint_{i_epoch:05d}_{index}.pt"), net, optimizer, i_epoch)

        for index, (net, optimizer) in enumerate(zip(nets, optimizers), start=1):
            save_model(os.path.join(c.MODEL_PATH_5, f"model_latest_{index}.pt"), net, optimizer, i_epoch, best_psnr)

        for scheduler in schedulers:
            scheduler.step()

    for index, (net, optimizer) in enumerate(zip(nets, optimizers), start=1):
        save_model(os.path.join(c.MODEL_PATH_5, f"model_final_{index}.pt"), net, optimizer, c.epochs, best_psnr)

#!/usr/bin/env python
import math
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

import config as c
import datasets
import modules.Unet_common as common
from model import Model_1, Model_2, Model_3, Model_4, Model_5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(path, net):
    state_dicts = torch.load(path, map_location=device, weights_only=False)
    network_state_dict = {key: value for key, value in state_dicts["net"].items() if "tmp_var" not in key}
    net.load_state_dict(network_state_dict)
    return int(state_dicts.get("epoch", 0))


def split_six(data):
    if data.shape[0] % 6 != 0:
        raise ValueError(f"The test batch size must be divisible by 6, but got {data.shape[0]}.")
    group_size = data.shape[0] // 6
    cover = data[:group_size]
    secrets = [data[i * group_size:(i + 1) * group_size] for i in range(1, 6)]
    return cover, secrets


def tensor_to_uint8_rgb(tensor):
    tensor = torch.clamp(tensor.detach(), 0.0, 1.0)
    tensor = tensor.mul(255.0).add(0.5).clamp(0.0, 255.0).to(torch.uint8)
    return tensor.permute(0, 2, 3, 1).cpu().numpy()


def bgr2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.0
    if only_y:
        result = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        result = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        result = result.round()
    else:
        result /= 255.0
    return result.astype(in_img_type)


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    if img1.ndim == 3 and img1.shape[2] == 1:
        return ssim(np.squeeze(img1), np.squeeze(img2))
    raise ValueError("PSNR and SSIM must be calculated on the Y channel.")


def calculate_mae(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mae = np.mean(np.abs(img1 - img2))
    if mae == 0:
        return float("inf")
    return mae


def calculate_rmse(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return np.sqrt(mse)


def update_metrics(metrics, name, reference, prediction):
    reference_rgb = tensor_to_uint8_rgb(reference)
    prediction_rgb = tensor_to_uint8_rgb(prediction)

    for reference_image, prediction_image in zip(reference_rgb, prediction_rgb):
        reference_bgr = reference_image[:, :, ::-1] / 255.0
        prediction_bgr = prediction_image[:, :, ::-1] / 255.0
        reference_y = bgr2ycbcr(reference_bgr) * 255.0
        prediction_y = bgr2ycbcr(prediction_bgr) * 255.0
        metrics[name]["psnr"].append(calculate_psnr(reference_y, prediction_y))
        metrics[name]["ssim"].append(calculate_ssim(reference_y, prediction_y))
        metrics[name]["mae"].append(calculate_mae(reference_image, prediction_image))
        metrics[name]["rmse"].append(calculate_rmse(reference_image, prediction_image))


def forward_cascade(nets, cover_dwt, secret_dwts, iwt):
    carrier_dwt = cover_dwt
    steg_dwts, steg_images = [], []

    for net, secret_dwt in zip(nets, secret_dwts):
        output_dwt, _ = net(torch.cat((carrier_dwt, secret_dwt), dim=1))
        steg_dwt = output_dwt.narrow(1, 0, 4 * c.channels_in)
        steg_dwts.append(steg_dwt)
        steg_images.append(iwt(steg_dwt))
        carrier_dwt = steg_dwt

    return steg_dwts, steg_images


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
    print("Start testing...")

    nets = [Model_1().to(device), Model_2().to(device), Model_3().to(device), Model_4().to(device), Model_5().to(device)]
    if torch.cuda.is_available():
        nets = [torch.nn.DataParallel(net, device_ids=c.device_ids) for net in nets]

    checkpoint_epochs = [load_model(os.path.join(c.MODEL_PATH_5, f"model_best_{index}.pt"), net) for index, net in enumerate(nets, start=1)]
    if len(set(checkpoint_epochs)) != 1:
        raise RuntimeError(f"The five checkpoints have different epochs: {checkpoint_epochs}.")

    for net in nets:
        net.eval()

    dwt = common.DWT()
    iwt = common.IWT()
    metric_names = [f"C-S{i}" for i in range(1, 6)] + [f"S{i}-R{i}" for i in range(1, 6)]
    metrics = {name: {"psnr": [], "ssim": [], "mae": [], "rmse": []} for name in metric_names}

    with torch.inference_mode():
        loop = tqdm(datasets.testloader, desc="Testing", leave=True)

        for data in loop:
            data = data.to(device)
            cover, secrets = split_six(data)
            cover_dwt = dwt(cover)
            secret_dwts = [dwt(secret) for secret in secrets]
            steg_dwts, steg_images = forward_cascade(nets, cover_dwt, secret_dwts, iwt)
            recovered_secrets = reverse_cascade(nets, steg_dwts[-1], iwt)

            for index in range(5):
                update_metrics(metrics, f"C-S{index + 1}", cover, steg_images[index])
                update_metrics(metrics, f"S{index + 1}-R{index + 1}", secrets[index], recovered_secrets[index])

            loop.set_postfix(C_S5=f"{np.mean(metrics['C-S5']['psnr']):.2f}", S1_R1=f"{np.mean(metrics['S1-R1']['psnr']):.2f}", S2_R2=f"{np.mean(metrics['S2-R2']['psnr']):.2f}", S3_R3=f"{np.mean(metrics['S3-R3']['psnr']):.2f}", S4_R4=f"{np.mean(metrics['S4-R4']['psnr']):.2f}", S5_R5=f"{np.mean(metrics['S5-R5']['psnr']):.2f}")

    print(f"\nCheckpoint epoch: {checkpoint_epochs[0]}")
    print("================ Test Results ================")

    for name in metric_names:
        avg_psnr = float(np.mean(metrics[name]["psnr"]))
        avg_ssim = float(np.mean(metrics[name]["ssim"]))
        avg_mae = float(np.mean(metrics[name]["mae"]))
        avg_rmse = float(np.mean(metrics[name]["rmse"]))
        print(f"{name}: PSNR-Y = {avg_psnr:.6f} dB | SSIM-Y = {avg_ssim:.6f} | MAE-RGB = {avg_mae:.6f} | RMSE-RGB = {avg_rmse:.6f}")

    print("==============================================")
    print("Finished testing.")

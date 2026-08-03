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
from model import Model_1, Model_2, Model_3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(path, net):
    state_dicts = torch.load(path, map_location=device, weights_only=False)
    network_state_dict = {key: value for key, value in state_dicts["net"].items() if "tmp_var" not in key}
    net.load_state_dict(network_state_dict)
    return int(state_dicts.get("epoch", 0))


def split_four(data):
    if data.shape[0] % 4 != 0:
        raise ValueError(f"The test batch size must be divisible by 4, but got {data.shape[0]}.")
    group_size = data.shape[0] // 4
    cover = data[:group_size]
    secret_1 = data[group_size:2 * group_size]
    secret_2 = data[2 * group_size:3 * group_size]
    secret_3 = data[3 * group_size:4 * group_size]
    return cover, secret_1, secret_2, secret_3


def tensor_to_uint8_rgb(tensor):
    tensor = torch.clamp(tensor.detach(), 0.0, 1.0)
    tensor = tensor.mul(255.0).add(0.5).clamp(0.0, 255.0).to(torch.uint8)
    return tensor.permute(0, 2, 3, 1).cpu().numpy()


def bgr2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img = img.astype(np.float32)
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
        reference_bgr = reference_image[:, :, ::-1].astype(np.float32) / 255.0
        prediction_bgr = prediction_image[:, :, ::-1].astype(np.float32) / 255.0
        reference_y = bgr2ycbcr(reference_bgr) * 255.0
        prediction_y = bgr2ycbcr(prediction_bgr) * 255.0

        metrics[name]["psnr"].append(calculate_psnr(reference_y, prediction_y))
        metrics[name]["ssim"].append(calculate_ssim(reference_y, prediction_y))
        metrics[name]["mae"].append(calculate_mae(reference_image, prediction_image))
        metrics[name]["rmse"].append(calculate_rmse(reference_image, prediction_image))


if __name__ == "__main__":
    print("Start testing...")

    net1 = Model_1().to(device)
    net2 = Model_2().to(device)
    net3 = Model_3().to(device)

    if torch.cuda.is_available():
        net1 = torch.nn.DataParallel(net1, device_ids=c.device_ids)
        net2 = torch.nn.DataParallel(net2, device_ids=c.device_ids)
        net3 = torch.nn.DataParallel(net3, device_ids=c.device_ids)

    epoch_1 = load_model(os.path.join(c.MODEL_PATH_3, "model_best_1.pt"), net1)
    epoch_2 = load_model(os.path.join(c.MODEL_PATH_3, "model_best_2.pt"), net2)
    epoch_3 = load_model(os.path.join(c.MODEL_PATH_3, "model_best_3.pt"), net3)

    if epoch_1 != epoch_2 or epoch_1 != epoch_3:
        raise RuntimeError(f"The three checkpoints have different epochs: {epoch_1}, {epoch_2} and {epoch_3}.")

    net1.eval()
    net2.eval()
    net3.eval()

    dwt = common.DWT()
    iwt = common.IWT()
    metric_names = ["C-S1", "C-S2", "C-S3", "S1-R1", "S2-R2", "S3-R3"]
    metrics = {name: {"psnr": [], "ssim": [], "mae": [], "rmse": []} for name in metric_names}

    with torch.inference_mode():
        loop = tqdm(datasets.testloader, desc="Testing", leave=True)

        for data in loop:
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

            reverse_dwt_3 = net3(output_steg_dwt_3, rev=True)
            recovered_steg_dwt_2 = reverse_dwt_3.narrow(1, 0, 4 * c.channels_in)
            recovered_secret_3 = iwt(reverse_dwt_3.narrow(1, 4 * c.channels_in, reverse_dwt_3.shape[1] - 4 * c.channels_in))

            reverse_dwt_2 = net2(recovered_steg_dwt_2, rev=True)
            recovered_steg_dwt_1 = reverse_dwt_2.narrow(1, 0, 4 * c.channels_in)
            recovered_secret_2 = iwt(reverse_dwt_2.narrow(1, 4 * c.channels_in, reverse_dwt_2.shape[1] - 4 * c.channels_in))

            reverse_dwt_1 = net1(recovered_steg_dwt_1, rev=True)
            recovered_secret_1 = iwt(reverse_dwt_1.narrow(1, 4 * c.channels_in, reverse_dwt_1.shape[1] - 4 * c.channels_in))

            update_metrics(metrics, "C-S1", cover, output_steg_1)
            update_metrics(metrics, "C-S2", cover, output_steg_2)
            update_metrics(metrics, "C-S3", cover, output_steg_3)
            update_metrics(metrics, "S1-R1", secret_1, recovered_secret_1)
            update_metrics(metrics, "S2-R2", secret_2, recovered_secret_2)
            update_metrics(metrics, "S3-R3", secret_3, recovered_secret_3)

            loop.set_postfix(C_S3=f"{np.mean(metrics['C-S3']['psnr']):.2f}", S1_R1=f"{np.mean(metrics['S1-R1']['psnr']):.2f}", S2_R2=f"{np.mean(metrics['S2-R2']['psnr']):.2f}", S3_R3=f"{np.mean(metrics['S3-R3']['psnr']):.2f}")

    print(f"\nCheckpoint epoch: {epoch_1}")
    print("================ Test Results ================")

    for name in metric_names:
        avg_psnr = float(np.mean(metrics[name]["psnr"]))
        avg_ssim = float(np.mean(metrics[name]["ssim"]))
        avg_mae = float(np.mean(metrics[name]["mae"]))
        avg_rmse = float(np.mean(metrics[name]["rmse"]))
        print(f"{name}: PSNR-Y = {avg_psnr:.6f} dB | SSIM-Y = {avg_ssim:.6f} | MAE-RGB = {avg_mae:.6f} | RMSE-RGB = {avg_rmse:.6f}")

    print("==============================================")
    print("Finished testing.")

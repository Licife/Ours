import os
from pathlib import Path

import torch
import torchvision

from model import Model_1, Model_2, Model_3, Model_4, init_model
import config as c
import datasets
import modules.Unet_common as common


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 可选：model_latest、model_final 或 model_best
CHECKPOINT_PREFIX = "model_best"

# 所有测试图像保存到该目录
IMAGE_PATH_4 = Path("image4")

SAVE_DIRS = {
    "cover": IMAGE_PATH_4 / "cover",
    "secret_1": IMAGE_PATH_4 / "secret_1",
    "secret_2": IMAGE_PATH_4 / "secret_2",
    "secret_3": IMAGE_PATH_4 / "secret_3",
    "secret_4": IMAGE_PATH_4 / "secret_4",
    "steg_1": IMAGE_PATH_4 / "steg_1",
    "steg_2": IMAGE_PATH_4 / "steg_2",
    "steg_3": IMAGE_PATH_4 / "steg_3",
    "steg_4": IMAGE_PATH_4 / "steg_4",
    "secret_rev_1": IMAGE_PATH_4 / "secret_rev_1",
    "secret_rev_2": IMAGE_PATH_4 / "secret_rev_2",
    "secret_rev_3": IMAGE_PATH_4 / "secret_rev_3",
    "secret_rev_4": IMAGE_PATH_4 / "secret_rev_4",
    "cover_rev": IMAGE_PATH_4 / "cover_rev",
}


def load_model(path, net):
    print("Loading checkpoint:", path)
    state_dicts = torch.load(path, map_location=device, weights_only=False)
    network_state_dict = {key: value for key, value in state_dicts["net"].items() if "tmp_var" not in key}
    net.load_state_dict(network_state_dict)
    return int(state_dicts.get("epoch", 0))


def split_five(data):
    if data.shape[0] % 5 != 0:
        raise ValueError(f"The test batch size must be divisible by 5, but got {data.shape[0]}.")

    group_size = data.shape[0] // 5
    cover = data[:group_size]
    secret_1 = data[group_size:2 * group_size]
    secret_2 = data[2 * group_size:3 * group_size]
    secret_3 = data[3 * group_size:4 * group_size]
    secret_4 = data[4 * group_size:5 * group_size]
    return cover, secret_1, secret_2, secret_3, secret_4


def save_batch(images, folder, start_index):
    images = torch.clamp(images.detach(), 0.0, 1.0)

    for batch_index in range(images.shape[0]):
        image_index = start_index + batch_index
        save_path = SAVE_DIRS[folder] / f"{image_index:05d}.png"
        torchvision.utils.save_image(images[batch_index:batch_index + 1], str(save_path))


if __name__ == "__main__":
    print("Start testing...")

    for directory in SAVE_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)

    net1 = Model_1().to(device)
    net2 = Model_2().to(device)
    net3 = Model_3().to(device)
    net4 = Model_4().to(device)

    init_model(net1)
    init_model(net2)
    init_model(net3)
    init_model(net4)

    if torch.cuda.is_available():
        net1 = torch.nn.DataParallel(net1, device_ids=c.device_ids)
        net2 = torch.nn.DataParallel(net2, device_ids=c.device_ids)
        net3 = torch.nn.DataParallel(net3, device_ids=c.device_ids)
        net4 = torch.nn.DataParallel(net4, device_ids=c.device_ids)

    epoch_1 = load_model(os.path.join(c.MODEL_PATH_4, f"{CHECKPOINT_PREFIX}_1.pt"), net1)
    epoch_2 = load_model(os.path.join(c.MODEL_PATH_4, f"{CHECKPOINT_PREFIX}_2.pt"), net2)
    epoch_3 = load_model(os.path.join(c.MODEL_PATH_4, f"{CHECKPOINT_PREFIX}_3.pt"), net3)
    epoch_4 = load_model(os.path.join(c.MODEL_PATH_4, f"{CHECKPOINT_PREFIX}_4.pt"), net4)

    if len({epoch_1, epoch_2, epoch_3, epoch_4}) != 1:
        raise RuntimeError(f"The four checkpoints have different epochs: {epoch_1}, {epoch_2}, {epoch_3}, {epoch_4}.")

    print("Checkpoint epoch:", epoch_1)

    net1.eval()
    net2.eval()
    net3.eval()
    net4.eval()

    dwt = common.DWT()
    iwt = common.IWT()
    image_counter = 0

    with torch.no_grad():
        for _, data in enumerate(datasets.testloader):
            data = data.to(device)
            cover, secret_1, secret_2, secret_3, secret_4 = split_five(data)

            cover_dwt = dwt(cover)
            secret_dwt_1 = dwt(secret_1)
            secret_dwt_2 = dwt(secret_2)
            secret_dwt_3 = dwt(secret_3)
            secret_dwt_4 = dwt(secret_4)

            # Stage 1: C + S1 -> I1
            output_dwt_1, _ = net1(torch.cat((cover_dwt, secret_dwt_1), dim=1))
            steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)
            steg_1 = iwt(steg_dwt_1)

            # Stage 2: I1 + S2 -> I2
            output_dwt_2, _ = net2(torch.cat((steg_dwt_1, secret_dwt_2), dim=1))
            steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)
            steg_2 = iwt(steg_dwt_2)

            # Stage 3: I2 + S3 -> I3
            output_dwt_3, _ = net3(torch.cat((steg_dwt_2, secret_dwt_3), dim=1))
            steg_dwt_3 = output_dwt_3.narrow(1, 0, 4 * c.channels_in)
            steg_3 = iwt(steg_dwt_3)

            # Stage 4: I3 + S4 -> I4
            output_dwt_4, _ = net4(torch.cat((steg_dwt_3, secret_dwt_4), dim=1))
            steg_dwt_4 = output_dwt_4.narrow(1, 0, 4 * c.channels_in)
            steg_4 = iwt(steg_dwt_4)

            # Stage 4 inverse: I4 -> RI3 + R4
            reverse_dwt_4 = net4(steg_dwt_4, rev=True)
            recovered_steg_dwt_3 = reverse_dwt_4.narrow(1, 0, 4 * c.channels_in)
            recovered_secret_dwt_4 = reverse_dwt_4.narrow(1, 4 * c.channels_in, reverse_dwt_4.shape[1] - 4 * c.channels_in)
            recovered_secret_4 = iwt(recovered_secret_dwt_4)

            # Stage 3 inverse: RI3 -> RI2 + R3
            reverse_dwt_3 = net3(recovered_steg_dwt_3, rev=True)
            recovered_steg_dwt_2 = reverse_dwt_3.narrow(1, 0, 4 * c.channels_in)
            recovered_secret_dwt_3 = reverse_dwt_3.narrow(1, 4 * c.channels_in, reverse_dwt_3.shape[1] - 4 * c.channels_in)
            recovered_secret_3 = iwt(recovered_secret_dwt_3)

            # Stage 2 inverse: RI2 -> RI1 + R2
            reverse_dwt_2 = net2(recovered_steg_dwt_2, rev=True)
            recovered_steg_dwt_1 = reverse_dwt_2.narrow(1, 0, 4 * c.channels_in)
            recovered_secret_dwt_2 = reverse_dwt_2.narrow(1, 4 * c.channels_in, reverse_dwt_2.shape[1] - 4 * c.channels_in)
            recovered_secret_2 = iwt(recovered_secret_dwt_2)

            # Stage 1 inverse: RI1 -> RC + R1
            reverse_dwt_1 = net1(recovered_steg_dwt_1, rev=True)
            recovered_cover_dwt = reverse_dwt_1.narrow(1, 0, 4 * c.channels_in)
            recovered_secret_dwt_1 = reverse_dwt_1.narrow(1, 4 * c.channels_in, reverse_dwt_1.shape[1] - 4 * c.channels_in)
            recovered_cover = iwt(recovered_cover_dwt)
            recovered_secret_1 = iwt(recovered_secret_dwt_1)

            batch_size = cover.shape[0]

            save_batch(cover, "cover", image_counter)
            save_batch(secret_1, "secret_1", image_counter)
            save_batch(secret_2, "secret_2", image_counter)
            save_batch(secret_3, "secret_3", image_counter)
            save_batch(secret_4, "secret_4", image_counter)

            save_batch(steg_1, "steg_1", image_counter)
            save_batch(steg_2, "steg_2", image_counter)
            save_batch(steg_3, "steg_3", image_counter)
            save_batch(steg_4, "steg_4", image_counter)

            save_batch(recovered_secret_1, "secret_rev_1", image_counter)
            save_batch(recovered_secret_2, "secret_rev_2", image_counter)
            save_batch(recovered_secret_3, "secret_rev_3", image_counter)
            save_batch(recovered_secret_4, "secret_rev_4", image_counter)
            save_batch(recovered_cover, "cover_rev", image_counter)

            image_counter += batch_size
            print(f"Saved {image_counter} image group(s).")

    print("Finished testing.")
    print("Images saved to:", IMAGE_PATH_4.resolve())

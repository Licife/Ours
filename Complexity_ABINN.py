# Complexity_ABINN_same_as_LiDiNet.py
# Measure Params, Hiding FLOPs, Full Forward FLOPs (Hide+Reveal), Training FLOPs,
# GPU Memory, Training Time, Testing Time, Total Time and FPS for ABINN.

import torch
import torch.nn as nn
from thop import profile
import random
import numpy as np

seed = 2026

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def strip_module_prefix(state_dict):
    """Remove 'module.' prefix saved by nn.DataParallel."""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state[k] = v
    return new_state


def load_checkpoint(model, ckpt_path):
    """Load common checkpoint formats. Safe for DataParallel checkpoints."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "net" in checkpoint:
        state_dict = checkpoint["net"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = strip_module_prefix(state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", len(msg.missing_keys))
    print("Unexpected keys:", len(msg.unexpected_keys))
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return total, trainable


def tensor_sum(output):
    """Create a dummy scalar loss for tuple/list/tensor outputs."""
    if torch.is_tensor(output):
        return output.sum()
    if isinstance(output, (tuple, list)):
        loss = 0.0
        for item in output:
            loss = loss + tensor_sum(item)
        return loss
    raise TypeError(f"Unsupported output type: {type(output)}")


def measure_forward_flops(model, cover, secret):
    model.eval()
    flops, _ = profile(model, inputs=(cover, secret), verbose=False)
    return flops / 1e9


def measure_training_time(model, cover, secret, optimizer, repeat=50, warmup=10):
    """One training step time: forward + backward + optimizer.step()."""
    model.train()

    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        out = model(cover, secret)
        loss = tensor_sum(out)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        times = []
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(repeat):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            starter.record()

            out = model(cover, secret)
            loss = tensor_sum(out)
            loss.backward()
            optimizer.step()

            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))

        return sum(times) / len(times)

    # CPU fallback
    import time
    times = []
    for _ in range(repeat):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        out = model(cover, secret)
        loss = tensor_sum(out)
        loss.backward()
        optimizer.step()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return sum(times) / len(times)


def measure_testing_time(model, cover, secret, repeat=100, warmup=10):
    """One testing/inference time: full forward under torch.no_grad()."""
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(cover, secret)

    if device.type == "cuda":
        times = []
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for _ in range(repeat):
                torch.cuda.synchronize()
                starter.record()
                _ = model(cover, secret)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))

        return sum(times) / len(times)

    # CPU fallback
    import time
    times = []
    with torch.no_grad():
        for _ in range(repeat):
            start = time.perf_counter()
            _ = model(cover, secret)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    return sum(times) / len(times)


def measure_memory(model, cover, secret):
    if device.type != "cuda":
        return 0.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.eval()

    with torch.no_grad():
        _ = model(cover, secret)

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 2


class ModeWrapper(nn.Module):
    """Wrap adapter with a fixed mode: hide or full."""
    def __init__(self, adapter, mode="full"):
        super().__init__()
        self.adapter = adapter
        self.mode = mode

    def forward(self, cover, secret):
        return self.adapter(cover, secret, mode=self.mode)


def complexity_test(
    adapter,
    image_size=256,
    train_repeat=50,
    test_repeat=100,
    outer_repeat=5,
):
    """
    Complexity test with 5 independent outer runs.

    - Params and FLOPs are deterministic for a fixed input size, so they are measured once.
    - GPU memory, training time, testing time, total time and FPS are measured over
      outer_repeat independent runs and then averaged.
    - Each outer run uses a newly generated random cover/secret pair.
    - Testing time and Forward FLOPs are measured by full_wrapper, i.e., Hide + Reveal.
    """
    hide_wrapper = ModeWrapper(adapter, mode="hide").to(device).eval()
    full_wrapper = ModeWrapper(adapter, mode="full").to(device).eval()

    # Use one random input pair to compute deterministic FLOPs.
    cover = torch.randn(1, 3, image_size, image_size).to(device)
    secret = torch.randn(1, 3, image_size, image_size).to(device)

    # 1) Hiding FLOPs: only hiding/concealing branch.
    hiding_flops = measure_forward_flops(hide_wrapper, cover, secret)

    # 2) Forward FLOPs: full testing path = hiding + revealing.
    forward_flops = measure_forward_flops(full_wrapper, cover, secret)

    # 3) Training FLOPs: theoretical estimate, forward + backward + update.
    training_flops = forward_flops * 3

    # 4) Params.
    total_params, trainable_params = count_params(adapter)

    memory_list = []
    train_time_list = []
    test_time_list = []
    total_time_list = []
    fps_list = []

    print("=" * 80)
    print(f"Outer Runs      : {outer_repeat}")
    print(f"Inner Train Rep : {train_repeat}")
    print(f"Inner Test Rep  : {test_repeat}")
    print("-" * 80)

    for run_idx in range(outer_repeat):
        # New random input pair for each outer run.
        cover = torch.randn(1, 3, image_size, image_size).to(device)
        secret = torch.randn(1, 3, image_size, image_size).to(device)

        # Peak memory for full testing path.
        memory = measure_memory(full_wrapper, cover, secret)

        # New optimizer for each outer run. This avoids optimizer state accumulation.
        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)

        train_time = measure_training_time(
            full_wrapper,
            cover,
            secret,
            optimizer,
            repeat=train_repeat,
        )

        test_time = measure_testing_time(
            full_wrapper,
            cover,
            secret,
            repeat=test_repeat,
        )

        total_time = train_time + test_time
        fps = 1000.0 / total_time

        memory_list.append(memory)
        train_time_list.append(train_time)
        test_time_list.append(test_time)
        total_time_list.append(total_time)
        fps_list.append(fps)

        print(
            f"Run {run_idx + 1:02d}/{outer_repeat} | "
            f"Memory: {memory:.2f} MB | "
            f"Training: {train_time:.3f} ms | "
            f"Testing: {test_time:.3f} ms | "
            f"Total: {total_time:.3f} ms | "
            f"FPS: {fps:.2f}"
        )

    memory_avg = sum(memory_list) / len(memory_list)
    train_time_avg = sum(train_time_list) / len(train_time_list)
    test_time_avg = sum(test_time_list) / len(test_time_list)
    total_time_avg = sum(total_time_list) / len(total_time_list)
    fps_avg = sum(fps_list) / len(fps_list)

    print("-" * 80)
    print("Average Results")
    print("=" * 80)
    print(f"Params          : {total_params:.3f} M")
    print(f"Trainable Params: {trainable_params:.3f} M")
    print(f"Hiding FLOPs    : {hiding_flops:.3f} G")
    print(f"Forward FLOPs   : {forward_flops:.3f} G")
    print(f"Training FLOPs  : {training_flops:.3f} G")
    print(f"GPU Memory      : {memory_avg:.2f} MB")
    print(f"Training Time   : {train_time_avg:.3f} ms")
    print(f"Testing Time    : {test_time_avg:.3f} ms")
    print(f"Total Time      : {total_time_avg:.3f} ms, FPS: {fps_avg:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    from model import Model
    import config as c
    import modules.Unet_common as common

    class ABINNAdapter(nn.Module):
        """
        Adapter for ABINN.

        mode='hide':
            cover + secret -> DWT -> ABINN forward -> stego_freq
            This corresponds to Hiding FLOPs.

        mode='full':
            cover + secret -> hiding -> stego_freq -> ABINN reverse -> recovered pair
            This corresponds to full testing FLOPs/time, i.e., hiding + revealing.
        """
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.dwt = common.DWT()
            self.channel = 4 * c.channels_in

        def forward(self, cover, secret, mode="full"):
            cover_input = self.dwt(cover)
            secret_input = self.dwt(secret)
            input_img = torch.cat((cover_input, secret_input), dim=1)

            output, _ = self.model(input_img, rev=False)
            output_steg = output.narrow(1, 0, self.channel)

            if mode == "hide":
                return output_steg

            if mode == "full":
                output_rev = self.model(output_steg, rev=True)
                return output, output_rev

            raise ValueError("mode must be 'hide' or 'full'")

    net = Model().to(device)

    # Optional: load checkpoint.
    # ckpt = c.MODEL_PATH + c.suffix
    # net = load_checkpoint(net, ckpt)

    adapter = ABINNAdapter(net).to(device)

    # Use 256 if your ABINN paper reports complexity at 256x256.
    # Use 128 if you want a fair comparison with LiHiNet/IIS reported at 128x128.
    complexity_test(adapter, image_size=128, outer_repeat=5)

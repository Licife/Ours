import os
import random
import shutil
from pathlib import Path

# ==================== 只需要修改这里 ====================
clean_folder = Path(r"D:\Reference_code\ISN\data\exp\DIV2K_Flickr2K_1multi_32c_8inv\results\cover")      # 原始载体图片文件夹
steg_folder = Path(r"D:\Reference_code\ISN\data\exp\DIV2K_Flickr2K_1multi_32c_8inv\results\steg")        # 隐写图片文件夹
output_folder = Path(r"D:\Reference_code\StegExpose\StegExpose\ISN_Data_Test")

clean_count = 2000                           # 需要抽取的载体图片数量
steg_count = 2000                            # 需要抽取的隐写图片数量
random_seed = 42                             # 固定随机种子，保证重复运行抽取结果一致
clear_output = True                          # True：运行前清空输出文件夹
# ======================================================

SUPPORTED_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


def get_images(folder):
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在：{folder}")
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])


def copy_and_rename(images, count, prefix, output_dir, rng):
    if len(images) < count:
        raise ValueError(f"{prefix} 图片数量不足：需要 {count} 张，实际只有 {len(images)} 张")

    selected = rng.sample(images, count)
    digits = max(4, len(str(count)))

    for index, src in enumerate(selected, start=1):
        dst = output_dir / f"{prefix}_{index:0{digits}d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    return selected


def main():
    rng = random.Random(random_seed)

    if clear_output and output_folder.exists():
        shutil.rmtree(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    clean_images = get_images(clean_folder)
    steg_images = get_images(steg_folder)

    selected_clean = copy_and_rename(clean_images, clean_count, "clean", output_folder, rng)
    selected_steg = copy_and_rename(steg_images, steg_count, "steg", output_folder, rng)

    record_path = output_folder / "selected_files.txt"
    with record_path.open("w", encoding="utf-8") as f:
        f.write("[clean]\n")
        for src in selected_clean:
            f.write(f"{src}\n")

        f.write("\n[steg]\n")
        for src in selected_steg:
            f.write(f"{src}\n")

    print("处理完成！")
    print(f"载体图片：{clean_count} 张，命名为 clean_XXXX")
    print(f"隐写图片：{steg_count} 张，命名为 steg_XXXX")
    print(f"输出目录：{output_folder}")
    print(f"抽取记录：{record_path}")


if __name__ == "__main__":
    main()

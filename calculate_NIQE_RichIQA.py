import os
import cv2
import glob
import torch
import numpy as np
from natsort import natsorted
import pyiqa


def main():
    # --- 1. 配置路径 ---
    # 对于无参考指标，我们只需要评估生成出来的图像 (Gen)
    folder_Gen = r'D:/Reference_code/Journal_End/image/steg_d'

    # 也可以用来评估恢复出的秘密图像
    # folder_Gen = r'D:/Reference_code/Journal_End/image/secret-rev_d'

    # 获取图像列表
    img_list = sorted(glob.glob(folder_Gen + '/*'))
    img_list = natsorted(img_list)

    # --- 2. 初始化 IQA 指标计算器 (借助 pyiqa) ---
    # 自动调用 GPU (如果有)，否则用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化 NIQE
    # NIQE 越低越好 (Lower is better)
    niqe_metric = pyiqa.create_metric('niqe').to(device)

    # 初始化深度主观质量评估模型
    # 注意：RichIQA 是一个具体的论文概念，如果 pyiqa 库中没有直接命名为 richiqa 的模型，
    # 学术界通常使用 MUSIQ、BRISQUE 或 CLIPIQA 来作为“丰富的深度主观感知”的替代指标。
    # 这里以最强大的主观感知深度模型 MUSIQ 为例 (支持多尺度，分数越高越好)
    try:
        deep_iqa_metric = pyiqa.create_metric('musiq').to(device)
    except:
        print("MUSIQ not found or failed to load, falling back to BRISQUE (another classic NR metric).")
        deep_iqa_metric = pyiqa.create_metric('brisque').to(device)

    niqe_all = []
    deep_iqa_all = []

    print(f'Starting evaluation on {len(img_list)} images...')

    # --- 3. 循环计算每张图的指标 ---
    for i, img_path in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # pyiqa 通常需要输入的张量格式为 (B, C, H, W)，取值范围 [0, 1]，RGB通道顺序
        # cv2 读取是 BGR，需转换为 RGB
        im_Gen_bgr = cv2.imread(img_path)
        if im_Gen_bgr is None:
            continue

        im_Gen_rgb = cv2.cvtColor(im_Gen_bgr, cv2.COLOR_BGR2RGB)

        # 转换为 PyTorch Tensor，归一化到 [0, 1]，并增加 Batch 维度
        im_tensor = torch.from_numpy(im_Gen_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        im_tensor = im_tensor.to(device)

        # 禁用梯度计算以加速推理
        with torch.no_grad():
            # 计算指标并提取数值
            score_niqe = niqe_metric(im_tensor).item()
            score_deep = deep_iqa_metric(im_tensor).item()

        niqe_all.append(score_niqe)
        deep_iqa_all.append(score_deep)

        print('{:3d} - {:25}. \tNIQE (↓): {:.4f}, \tDeep-IQA (↑/↓): {:.4f}'.format(
            i + 1, base_name, score_niqe, score_deep))

    # --- 4. 统计并输出结果 ---
    avg_niqe = sum(niqe_all) / len(niqe_all)
    avg_deep = sum(deep_iqa_all) / len(deep_iqa_all)

    print('\n=======================================')
    print('Average Results:')
    print('NIQE (Lower is better): {:.4f}'.format(avg_niqe))
    print('Deep-IQA Metric:        {:.4f}'.format(avg_deep))
    print('=======================================')

    # 将结果保存到 txt 文件
    # with open('NR_IQA_Results.txt', 'w') as f:
    #     f.write("NIQE Scores:\n")
    #     f.write(str(niqe_all) + "\n\n")
    #     f.write("Deep-IQA Scores:\n")
    #     f.write(str(deep_iqa_all) + "\n")


if __name__ == '__main__':
    main()
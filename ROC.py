import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

csv_folder = 'D:/Reference_code/StegExpose/StegExpose/'  # 替换为你的CSV文件所在的文件夹路径

csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

if not csv_files:
    print(f"在 {csv_folder} 目录下没有找到 CSV 文件，请检查路径！")
    exit()

# 初始化画布
plt.figure(figsize=(10, 8))

# 遍历每一个 CSV 文件
for file_path in csv_files:
    try:
        # 读取数据
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()  # 清理表头空格

        # 提取真实标签 (包含 'steg' 的为隐秘图像)
        # 注意：这里假设你的图片命名规则依然是 clean_xxx 和 steg_xxx
        y_true = df['File name'].astype(str).apply(lambda x: 1 if 'steg' in x.lower() else 0)

        # 提取预测得分
        y_scores = pd.to_numeric(df['Fusion (mean)'], errors='coerce')

        # 删除融合分数无效的样本
        valid_mask = y_scores.notna()
        y_true = y_true[valid_mask]
        y_scores = y_scores[valid_mask]

        # ROC 计算要求同时包含载体图像和隐写图像
        if y_true.nunique() < 2:
            print(f"跳过 {file_path}：该 CSV 中没有同时包含载体图像和隐写图像")
            continue

        # 自动计算 FPR 和 TPR
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        # 计算 AUC
        roc_auc = auc(fpr, tpr)

        # 获取不带后缀的文件名，用作图例的名称 (比如 'model_A')
        model_name = os.path.splitext(os.path.basename(file_path))[0]

        # 绘制该模型的曲线
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
        print(f"已处理: {model_name}, AUC: {roc_auc:.4f}")

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

# 绘制对角线 (随机猜测线)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing (AUC = 0.50)')

# 设置图表格式
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)

# 保存并显示
plt.savefig('comparison_roc_curve.png', dpi=300, bbox_inches='tight')
print("🎉 绘图完成！多模型对比 ROC 曲线已保存为 comparison_roc_curve.png")
plt.show()
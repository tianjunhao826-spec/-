import os
import shutil
import glob
import random
from tqdm import tqdm

# ================= 配置区域 =================

# --- 来源 A：旧的训练集 (需要降采样) ---
SRC_A_IMG = r"C:\Users\86153\ML_Projects\day01\毕设\yolo_dataset\images\train"
SRC_A_LBL = r"C:\Users\86153\ML_Projects\day01\毕设\yolo_dataset\labels\train"

# --- 来源 B：新数据 archive (1) (全保留，需要切分验证集) ---
SRC_B_IMG = r"C:\Users\86153\ML_Projects\day01\毕设\data\archive (1)\images"
# 确保这里是清洗后的标签路径
SRC_B_LBL = r"C:\Users\86153\ML_Projects\day01\毕设\data\archive (1)\labels_cleaned_final"

# --- 来源 C：旧的验证集 (全部保留，直接挪过去) ---
SRC_C_IMG = r"C:\Users\86153\ML_Projects\day01\毕设\yolo_dataset\images\val"
SRC_C_LBL = r"C:\Users\86153\ML_Projects\day01\毕设\yolo_dataset\labels\val"

# --- 输出位置 (新建文件夹，不覆盖原数据) ---
TARGET_ROOT = r"C:\Users\86153\ML_Projects\day01\毕设\yolo_dataset\dataset"

# --- 参数设置 ---
SAMPLE_STRIDE_A = 3
VAL_RATIO_B = 0.2  # 来源B的验证集比例


# ===========================================

def copy_file_pair(img_path, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    """复制图片和对应的标签文件的辅助函数"""
    filename = os.path.basename(img_path)
    label_name = os.path.splitext(filename)[0] + ".txt"

    # 1. 复制图片
    shutil.copy(img_path, os.path.join(dst_img_dir, filename))

    # 2. 复制标签
    src_label_path = os.path.join(src_lbl_dir, label_name)
    dst_label_path = os.path.join(dst_lbl_dir, label_name)

    if os.path.exists(src_label_path):
        shutil.copy(src_label_path, dst_label_path)
    else:
        # 如果没有标签文件，视情况而定，这里默认跳过标签复制
        pass


def build_mixed_dataset():
    # 1. 创建目录结构
    new_train_img = os.path.join(TARGET_ROOT, "images", "train")
    new_val_img = os.path.join(TARGET_ROOT, "images", "val")
    new_train_lbl = os.path.join(TARGET_ROOT, "labels", "train")
    new_val_lbl = os.path.join(TARGET_ROOT, "labels", "val")

    for d in [new_train_img, new_val_img, new_train_lbl, new_val_lbl]:
        os.makedirs(d, exist_ok=True)

    print(f"[INFO] 目标目录已创建: {TARGET_ROOT}")

    # ================= 步骤 1: 处理来源 A (旧训练集) =================
    print("\n[PROCESSING] 正在处理来源 A: 旧训练集 (执行降采样)...")
    imgs_a = sorted(glob.glob(os.path.join(SRC_A_IMG, "*.jpg")))

    if not imgs_a:
        print("[WARN] 来源 A 没有找到图片，请检查路径。")
    else:
        # 降采样：每 SAMPLE_STRIDE_A 张取 1 张
        sampled_a = imgs_a[::SAMPLE_STRIDE_A]
        print(f"   - 原数量: {len(imgs_a)} -> 降采样后: {len(sampled_a)}")

        # 来源 A -> 新训练集
        for img in tqdm(sampled_a, desc="Source A -> Train"):
            copy_file_pair(img, SRC_A_LBL, new_train_img, new_train_lbl)

    # ================= 步骤 2: 处理来源 C (旧验证集) =================
    # 注意：先处理旧验证集，确保这部分数据完整进入新验证集
    print("\n[PROCESSING] 正在处理来源 C: 旧验证集 (全部复制)...")
    imgs_c = glob.glob(os.path.join(SRC_C_IMG, "*.jpg"))

    if not imgs_c:
        print("[WARN] 来源 C (旧验证集) 没有找到图片。")
    else:
        print(f"   - 发现旧验证集图片数量: {len(imgs_c)}")
        for img in tqdm(imgs_c, desc="Source C -> Val"):
            copy_file_pair(img, SRC_C_LBL, new_val_img, new_val_lbl)

    # ================= 步骤 3: 处理来源 B (Archive新数据) =================
    print("\n[PROCESSING] 正在处理来源 B: 新增数据 (混合划分)...")
    imgs_b = []
    # 扫描多种格式
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        imgs_b.extend(glob.glob(os.path.join(SRC_B_IMG, ext)))

    if not imgs_b:
        print("[WARN] 来源 B 没有找到图片，请检查路径。")
    else:
        # 打乱顺序
        random.seed(42)
        random.shuffle(imgs_b)

        # 计算切分点
        val_count = int(len(imgs_b) * VAL_RATIO_B)

        val_set_b = imgs_b[:val_count]  # 20% 去验证集
        train_set_b = imgs_b[val_count:]  # 80% 去训练集

        print(f"   - 总数量: {len(imgs_b)}")
        print(f"   - 划分到验证集: {len(val_set_b)}")
        print(f"   - 划分到训练集: {len(train_set_b)}")

        # 复制到验证集
        for img in tqdm(val_set_b, desc="Source B -> Val"):
            copy_file_pair(img, SRC_B_LBL, new_val_img, new_val_lbl)

        # 复制到训练集
        for img in tqdm(train_set_b, desc="Source B -> Train"):
            copy_file_pair(img, SRC_B_LBL, new_train_img, new_train_lbl)

    # ================= 总结 =================
    print("\n" + "=" * 50)
    print("[DONE] 混合数据集构建完成")
    print(f"保存路径: {TARGET_ROOT}")

    # 简单的统计
    train_count = len(os.listdir(new_train_img))
    val_count = len(os.listdir(new_val_img))

    print(f"最终训练集图片数: {train_count}")
    print(f"最终验证集图片数: {val_count}")

if __name__ == "__main__":
    build_mixed_dataset()
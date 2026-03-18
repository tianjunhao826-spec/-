import os
import glob
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 原始标签文件夹路径
SOURCE_LABEL_DIR = r"C:\Users\86153\ML_Projects\day01\毕设\data\archive (1)\labels"

# 2. 清洗后的标签保存路径
# 建议新建一个文件夹，防止覆盖原始数据
TARGET_LABEL_DIR = r"C:\Users\86153\ML_Projects\day01\毕设\data\archive (1)\labels_cleaned_final"


# ===========================================

def convert_labels():
    # 如果输出目录不存在，则创建
    if not os.path.exists(TARGET_LABEL_DIR):
        os.makedirs(TARGET_LABEL_DIR)

    # 获取所有 txt 文件
    txt_files = glob.glob(os.path.join(SOURCE_LABEL_DIR, "*.txt"))

    if len(txt_files) == 0:
        print("❌ 未找到标签文件，请检查路径！")
        return

    print(f"🚀 开始处理 {len(txt_files)} 个文件...")
    print("🎯 目标: 移除原始ID 1，保留 Car(3->0), Bus(6->1), Van(8->2), 其余->Others(3)")

    # 统计计数器
    stats = {
        0: 0,  # Car
        1: 0,  # Bus
        2: 0,  # Van
        3: 0,  # Others
        "deleted_person": 0  # 统计删除了多少个"人"
    }

    for txt_file in tqdm(txt_files):
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:
            parts = line.strip().split()

            if len(parts) >= 5:
                raw_id = int(parts[0])
                coords = parts[1:]  # 坐标信息

                # ================= 核心逻辑 =================
                if raw_id == 1:
                    # 🔴 遇到原始标签 1 (人)，直接跳过，不写入新文件
                    stats["deleted_person"] += 1
                    continue

                elif raw_id == 3:
                    new_id = 0  # Car
                elif raw_id == 6:
                    new_id = 1  # Bus
                elif raw_id == 8:
                    new_id = 2  # Van
                else:
                    new_id = 3  # Others (除了1/3/6/8之外的所有)

                # 记录统计并添加到新列表
                stats[new_id] += 1
                new_line = f"{new_id} " + " ".join(coords) + "\n"
                new_lines.append(new_line)
                # ===========================================

        # 写入新文件
        file_name = os.path.basename(txt_file)
        save_path = os.path.join(TARGET_LABEL_DIR, file_name)

        with open(save_path, 'w') as f:
            f.writelines(new_lines)

    # 打印最终报告
    print("\n" + "=" * 40)
    print("✅ 数据清洗完成！")
    print("=" * 40)
    print(f"🚗 ID 0 (Car)    : {stats[0]}")
    print(f"🚌 ID 1 (Bus)    : {stats[1]}")
    print(f"🚛 ID 2 (Van)    : {stats[2]}")
    print(f"📦 ID 3 (Others) : {stats[3]}")
    print("-" * 40)
    print(f"🚫 已剔除原始 ID 1 (Person): {stats['deleted_person']} 个目标")
    print("=" * 40)
    print(f"📂 结果保存在: {TARGET_LABEL_DIR}")


if __name__ == "__main__":
    convert_labels()
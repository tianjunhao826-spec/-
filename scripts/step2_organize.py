import os
import shutil
import random
from tqdm import tqdm

# --- 核心配置 ---
# 这是一个列表，包含您存放图片的所有根目录
IMAGE_SOURCE_DIRS = [r'C:\Users\86153\ML_Projects\day01\毕设\data\Insight-MVT_Annotation_Train', r'C:\Users\86153\ML_Projects\day01\毕设\data\Insight-MVT_Annotation_Test']

# 标签来源（上一步生成的）
SOURCE_LABELS_DIR = r'C:\Users\86153\ML_Projects\day01\毕设\processed_data\output_labels'

# 最终输出位置
DEST_DIR = r'C:\Users\86153\ML_Projects\day01\毕设\yolo_dataset'
TRAIN_RATIO = 0.9


def setup_dirs():
    for category in ['images', 'labels']:
        for split in ['train', 'val']:
            path = os.path.join(DEST_DIR, category, split)
            if not os.path.exists(path):
                os.makedirs(path)


def organize():
    setup_dirs()

    # 获取 output_labels 下所有的序列名 (MVI_xxxxx)
    sequences = [d for d in os.listdir(SOURCE_LABELS_DIR) if os.path.isdir(os.path.join(SOURCE_LABELS_DIR, d))]

    print(f"总共找到 {len(sequences)} 个序列的标签，开始匹配图片...")

    count_success = 0
    count_fail = 0

    for seq_name in tqdm(sequences):
        label_seq_dir = os.path.join(SOURCE_LABELS_DIR, seq_name)

        # --- 关键逻辑：在两个图片文件夹里寻找该序列 ---
        img_seq_dir = None
        for source_dir in IMAGE_SOURCE_DIRS:
            potential_path = os.path.join(source_dir, seq_name)
            if os.path.exists(potential_path):
                img_seq_dir = potential_path
                break  # 找到了就停止寻找

        if img_seq_dir is None:
            # 如果两个文件夹里都找不到这个序列的图片
            print(f"警告：序列 {seq_name} 有标签，但没在 Train-Data 或 Test-Data 里找到对应的图片文件夹。")
            count_fail += 1
            continue

        # 处理该序列下的文件
        txt_files = [f for f in os.listdir(label_seq_dir) if f.endswith('.txt')]

        for txt_file in txt_files:
            base_name = os.path.splitext(txt_file)[0]
            img_file = base_name + '.jpg'

            src_txt_path = os.path.join(label_seq_dir, txt_file)
            src_img_path = os.path.join(img_seq_dir, img_file)

            if not os.path.exists(src_img_path):
                continue

            # 随机划分
            split = 'train' if random.random() < TRAIN_RATIO else 'val'

            # 构造唯一文件名 (防止重名)
            new_filename = f"{seq_name}_{base_name}"

            dst_img_path = os.path.join(DEST_DIR, 'images', split, new_filename + '.jpg')
            dst_txt_path = os.path.join(DEST_DIR, 'labels', split, new_filename + '.txt')

            shutil.copy(src_img_path, dst_img_path)
            shutil.copy(src_txt_path, dst_txt_path)
            count_success += 1

    print("\n处理完成！")
    print(f"成功处理图片数量: {count_success}")
    print(f"失败(缺失图片)序列数: {count_fail}")
    print(f"最终数据集位置: {os.path.abspath(DEST_DIR)}")


if __name__ == "__main__":
    organize()
from ultralytics import YOLO
import torch
import os
import yaml
import glob
import random


def create_mini_dataset_config(original_yaml_path, num_samples=500):
    """
    创建一个临时 yaml 配置，只包含少量样本，用于快速测试
    """
    # 1. 读取原始 yaml
    with open(original_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # 获取数据集根目录
    base_path = data.get('path', '')
    train_dir = os.path.join(base_path, data['train'])

    # 2. 找到所有训练图片
    # 支持 jpg, png, jpeg 等格式
    image_files = glob.glob(os.path.join(train_dir, '*.jpg')) + \
                  glob.glob(os.path.join(train_dir, '*.png'))

    if len(image_files) == 0:
        print("❌ 错误：在训练集中没找到图片，请检查路径！")
        return None

    print(f"[INFO] 原始训练集共有 {len(image_files)} 张图片")

    # 3. 随机抽取 num_samples 张图片（不移动文件，只生成包含路径的txt文件）
    # 如果总数少于抽取数，就用全部
    sample_count = min(len(image_files), num_samples)
    selected_images = random.sample(image_files, sample_count)

    # 4. 生成一个 txt 文件列表，作为新的训练源
    # YOLO 支持直接用 txt 文件列表代替文件夹路径
    mini_train_txt = os.path.join(base_path, 'mini_train_list.txt')
    with open(mini_train_txt, 'w', encoding='utf-8') as f:
        for img_path in selected_images:
            f.write(img_path + '\n')

    print(f"[INFO] 已生成临时训练列表: {mini_train_txt} (包含 {sample_count} 张图片)")

    # 5. 修改配置数据
    data['train'] = mini_train_txt  # 将训练路径指向这个 txt 文件

    # 6. 保存为新的临时 yaml
    mini_yaml_path = original_yaml_path.replace('.yaml', '_mini.yaml')
    with open(mini_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)

    print(f"[INFO] 临时迷你配置文件已生成: {mini_yaml_path}")
    return mini_yaml_path


def main():
    # =========================================================
    # 1. 路径定义
    # =========================================================
    weights_path = r'C:\Users\86153\ML_Projects\day01\毕设\weights\yolov10s.pt'
    original_yaml_path = r'C:\Users\86153\ML_Projects\day01\毕设\yolo_dataset\data.yaml'

    # =========================================================
    # 2. 自动生成迷你数据集配置 (关键步骤)
    # =========================================================
    # 这里设置抽取 500 张图片，你可以根据需要修改 num_samples
    print("[INFO] 正在准备迷你数据集以进行快速测试...")
    mini_yaml_path = create_mini_dataset_config(original_yaml_path, num_samples=500)

    if not mini_yaml_path:
        return

    # =========================================================
    # 3. 环境检查与模型加载
    # =========================================================
    device_type = 'GPU' if torch.cuda.is_available() else 'CPU'
    print(f"[INFO] Training device: {device_type}")

    if os.path.exists(weights_path):
        model = YOLO(weights_path)
        print(f"[INFO] Loaded local weights: {weights_path}")
    else:
        print(f"[WARNING] Local weights not found. Downloading yolov10s.pt...")
        model = YOLO('yolov10s.pt')

    # =========================================================
    # 4. 执行训练 (使用 mini_yaml_path)
    # =========================================================
    print(f"[INFO] Starting MINI-BATCH training experiment...")

    results = model.train(
        # --- 使用临时生成的迷你配置 ---
        data=mini_yaml_path,

        project="Traffic_Project",
        name="yolov10_mini_test",  # 改个名字，区分正式训练

        # --- 快速测试参数 ---
        epochs=10,  # 只要跑10轮看看能不能收敛
        batch=8,  # 显存安全值
        imgsz=640,
        device='0',
        workers=2,  # 少量数据用2个线程够了

        # --- 调参策略 ---
        optimizer='SGD',
        lr0=0.01,

        # --- 其他 ---
        save=True,
        plots=True,
        exist_ok=True
    )

    print(f"[INFO] Mini Training completed.")
    print(f"[INFO] 测试完成后，你可以检查: {results.save_dir}")
    print(f"[INFO] 原始数据毫发无损，下次正式训练只需把 data 参数换回原 yaml 即可。")


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
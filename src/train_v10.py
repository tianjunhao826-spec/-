from ultralytics import YOLO
import torch
import os


def main():
    # =========================================================
    # 1. 路径定义 (直接写在这里)
    # =========================================================
    weights_path = r'C:\Users\86153\ML_Projects\day01\毕设\weights\yolov10s.pt'
    data_yaml_path = r'C:\Users\86153\ML_Projects\day01\毕设\yolo_dataset\data.yaml'

    # =========================================================
    # 2. 环境检查与模型加载
    # =========================================================
    device_type = 'GPU' if torch.cuda.is_available() else 'CPU'
    print(f"[INFO] Training device: {device_type}")

    # 优先加载本地权重
    if os.path.exists(weights_path):
        model = YOLO(weights_path)
        print(f"[INFO] Loaded local weights: {weights_path}")
    else:
        model = YOLO('yolov10s.pt')

    # 3. 执行训练
    print(f"[INFO] Starting training experiment...")

    results = model.train(
        # --- 基础配置 ---
        data=data_yaml_path,
        project="Traffic_Project",
        name="yolov10_train_v1",

        # --- 训练参数 ---
        epochs=50,  # 毕设演示50轮足够
        batch=8,  # 显存优化值
        imgsz=640,
        cache='disk',
        device='0',  # 指定GPU 0`.
        workers=4,  # 开启4线程加速数据加载
        amp=True,
        # --- 调参策略 ---
        patience=15,  # 早停
        optimizer='SGD',  # 优化器
        lr0=0.01,  # 初始学习率
        cos_lr=True,  # 余弦退火

        # --- 其他 ---
        save=True,  # 保存模型
        plots=True,  # 绘制图表
        exist_ok=True  # 允许覆盖
    )

    print(f"[INFO] Training completed.")
    print(f"[INFO] Best model saved at: {results.save_dir}\\weights\\best.pt")


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
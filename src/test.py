from ultralytics import YOLO
import os
import glob

# ==============================================================================
# 测试配置 (Configuration)
# ==============================================================================
# 指向训练生成的最佳权重文件
MODEL_PATH = r"Traffic_Project\yolov10_train_v1\weights\best.pt"
DATA_YAML = r"C:\Users\86153\ML_Projects\day01\毕设\yolo_dataset\data.yaml"
# 用于可视化的测试图片目录 (通常使用验证集)
TEST_IMAGES_DIR = r"C:\Users\86153\ML_Projects\day01\毕设\yolo_dataset\dataset\images\val"


def evaluate_model():
    # 1. 模型文件校验
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("Please run train_main.py first.")
        return

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # ---------------------------------------------------------
    # 阶段 A: 定量评估 (Quantitative Evaluation)
    # ---------------------------------------------------------
    print("[INFO] Starting validation on dataset...")
    metrics = model.val(
        data=DATA_YAML,
        split='val',
        imgsz=640,
        batch=16,
        conf=0.001,  # 低置信度以计算完整PR曲线
        plots=True
    )

    # 提取关键指标
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr

    print("\n" + "=" * 50)
    print("TEST REPORT: MODEL PERFORMANCE METRICS")
    print("=" * 50)
    print(f"{'Metric':<20} | {'Value':<10}")
    print("-" * 34)
    print(f"{'mAP@0.5':<20} | {map50:.4f}")
    print(f"{'mAP@0.5:0.95':<20} | {map50_95:.4f}")
    print(f"{'Precision':<20} | {precision:.4f}")
    print(f"{'Recall':<20} | {recall:.4f}")
    print("=" * 50)
    print(f"[INFO] Validation plots saved to: {metrics.save_dir}")

    # ---------------------------------------------------------
    # 阶段 B: 可视化测试 (Qualitative Visualization)
    # ---------------------------------------------------------
    print("\n[INFO] Starting visualization test...")

    # 获取前 5 张测试图片
    images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))[:5]

    if not images:
        print("[WARNING] No images found in test directory.")
        return

    # 执行推理
    results = model.predict(
        source=images,
        save=True,
        conf=0.25,  # 置信度阈值
        iou=0.45,  # NMS 阈值
        line_width=2,  # 绘图线宽
        project="Traffic_Project",
        name="test_results"
    )

    save_dir = results[0].save_dir
    print(f"[INFO] Visualization results saved to: {save_dir}")


if __name__ == '__main__':
    evaluate_model()
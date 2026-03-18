import cv2
import os
import glob
import random
from ultralytics import YOLO

# ==========================================
# 配置区域
# ==========================================
# 1. 模型路径：指向你刚才训练出的最佳权重
MODEL_PATH = r'C:\Users\86153\ML_Projects\day01\毕设\src\Traffic_Project\yolov10_mini_test\weights\best.pt'

# 2. 测试图片来源：使用原本的验证集文件夹
TEST_IMAGE_DIR = r'C:\Users\86153\ML_Projects\day01\毕设\yolo_dataset\images\val'

# 3. 输出保存位置
OUTPUT_DIR = r'C:\Users\86153\ML_Projects\day01\毕设\output\inference_results'


def predict_images():
    """从验证集中随机抽取图片进行推理并保存"""
    print(f"[INFO] 加载模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("❌ 错误：找不到模型文件，请检查路径！")
        return

    model = YOLO(MODEL_PATH)

    # 获取所有 jpg 图片
    image_files = glob.glob(os.path.join(TEST_IMAGE_DIR, "*.jpg"))
    if not image_files:
        print("❌ 错误：测试目录下没有图片！")
        return

    # 随机抽取 5 张看效果
    num_samples = 5
    samples = random.sample(image_files, min(len(image_files), num_samples))

    print(f"[INFO] 正在对 {len(samples)} 张图片进行推理...")

    # 执行推理
    # save=True 会自动保存结果到 runs/detect/predict...
    # 但为了方便管理，我们手动指定 project 和 name
    results = model.predict(
        source=samples,
        save=True,
        conf=0.25,  # 置信度阈值，低于0.25的不显示
        iou=0.45,  # NMS阈值
        project=OUTPUT_DIR,
        name='batch_test',
        exist_ok=True
    )

    print(f"✅ 图片推理完成！")
    print(f"📂 结果已保存在: {results[0].save_dir}")
    print(f"   (请打开文件夹查看画了框的图片)")


def predict_video(video_path=None):
    """
    视频流推理演示
    如果不传 video_path，默认尝试打开摄像头(0)
    """
    model = YOLO(MODEL_PATH)

    source = video_path if video_path else 0
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("❌ 无法打开摄像头或视频文件")
        return

    print("🚀 按 'q' 键退出播放")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 推理
        # stream=True 对于视频流更高效
        results = model.predict(frame, conf=0.4, verbose=False)

        # 在帧上画框
        annotated_frame = results[0].plot()

        # 显示
        cv2.imshow("YOLOv10 Inference Demo", annotated_frame)

        # 按 Q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # --- 模式选择 ---

    # 模式 1: 图片测试 (毕设论文贴图用这个)
    predict_images()

    # 模式 2: 视频测试 (答辩演示用这个)
    # 如果你有视频文件，把下面这行解注释，并填入视频路径
    # predict_video(r'C:\Users\86153\Downloads\traffic_test.mp4')

    # 如果想测试笔记本摄像头，直接运行这个：
    # predict_video()
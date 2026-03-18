import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

# --- 配置区域 ---
IMG_WIDTH = 960
IMG_HEIGHT = 540

# 类别映射
CLASS_MAPPING = {
    'car': 0, 'bus': 1, 'van': 2, 'others': 3
}

# 输出的总文件夹
OUTPUT_DIR = r'C:\Users\86153\ML_Projects\day01\毕设\processed_data/output_labels'


def convert_box(box_info):
    left = float(box_info.attrib['left'])
    top = float(box_info.attrib['top'])
    width = float(box_info.attrib['width'])
    height = float(box_info.attrib['height'])
    center_x = left + (width / 2)
    center_y = top + (height / 2)
    x = max(0, min(1, center_x / IMG_WIDTH))
    y = max(0, min(1, center_y / IMG_HEIGHT))
    w = max(0, min(1, width / IMG_WIDTH))
    h = max(0, min(1, height / IMG_HEIGHT))
    return x, y, w, h


def process_xml_files(xml_dir, output_root):
    # 确保输出目录存在
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 获取所有 XML 文件
    if not os.path.exists(xml_dir):
        print(f"报错：找不到文件夹 {xml_dir}")
        return

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"正在处理文件夹: {xml_dir}，共 {len(xml_files)} 个文件...")

    for xml_file in tqdm(xml_files):
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()
        seq_name = root.attrib['name']

        # 为每个序列创建子文件夹
        seq_out_dir = os.path.join(output_root, seq_name)
        if not os.path.exists(seq_out_dir):
            os.makedirs(seq_out_dir)

        for frame in root.findall('frame'):
            frame_num = int(frame.attrib['num'])
            txt_filename = f"img{frame_num:05d}.txt"
            txt_path = os.path.join(seq_out_dir, txt_filename)

            target_list = frame.find('target_list')
            if target_list is None: continue

            with open(txt_path, 'w') as f:
                for target in target_list.findall('target'):
                    attribute = target.find('attribute')
                    # 有些测试集可能没有 vehicle_type 属性，做个防报错处理
                    if attribute is not None:
                        vehicle_type = attribute.attrib.get('vehicle_type', 'others')
                    else:
                        vehicle_type = 'car'  # 默认值

                    class_id = CLASS_MAPPING.get(vehicle_type, 3)
                    box = target.find('box')
                    if box is not None:
                        bb = convert_box(box)
                        f.write(f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")


if __name__ == "__main__":
    # 1. 处理训练集 XML
    process_xml_files(r'C:\Users\86153\ML_Projects\day01\毕设\data\DETRAC-Train-Annotations-XML', OUTPUT_DIR)

    # 2. 处理测试集 XML (追加到同一个输出文件夹)
    process_xml_files(r'C:\Users\86153\ML_Projects\day01\毕设\data\DETRAC-Test-Annotations-XML', OUTPUT_DIR)

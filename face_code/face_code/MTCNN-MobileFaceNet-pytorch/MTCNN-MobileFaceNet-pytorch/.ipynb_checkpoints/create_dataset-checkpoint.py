import os
import struct
import uuid
from tqdm import tqdm
import cv2

class DataSetWriter:
    def __init__(self, prefix):
        self.data_file = open(prefix + '.data', 'wb')
        self.header_file = open(prefix + '.header', 'wb')
        self.label_file = open(prefix + '.label', 'wb')
        self.offset = 0

    def add_img(self, key, img):
        self.data_file.write(struct.pack('I', len(key)))
        self.data_file.write(key.encode('ascii'))
        self.data_file.write(struct.pack('I', len(img)))
        self.data_file.write(img)
        self.offset += 4 + len(key) + 4
        self.header_file.write(f"{key}\t{self.offset}\t{len(img)}\n".encode('ascii'))
        self.offset += len(img)

    def add_label(self, label):
        self.label_file.write(label.encode('ascii') + b'\n')

def convert_data(root_path, output_prefix):
    data = []
    # 获取所有人物目录并排序（确保person01, person02...的顺序）
    person_dirs = sorted([d for d in os.listdir(root_path) if d.startswith('person')])
    
    for person_id, person_dir in enumerate(person_dirs):  # 使用enumerate自动生成person_id
        person_path = os.path.join(root_path, person_dir)
        if not os.path.isdir(person_path):
            continue
        for image in os.listdir(person_path):
            image_path = os.path.join(person_path, image)
            data.append((image_path, person_id))  # person_id从0开始递增
    
    print(f"训练数据大小：{len(data)}，总类别为：{len(person_dirs)}")

    writer = DataSetWriter(output_prefix)
    for image_path, person_id in tqdm(data):
        try:
            key = str(uuid.uuid1())
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            _, img_encoded = cv2.imencode('.bmp', img)
            writer.add_img(key, img_encoded.tobytes())
            writer.add_label(f"{key}\t{person_id}")
        except Exception as e:
            print(f"跳过文件 {image_path}: {str(e)}")
            continue

if __name__ == '__main__':
    # load_mx_rec(Path('dataset'), Path('dataset/faces_emore'))
    convert_data('dataset/train_data', 'dataset/train_data')
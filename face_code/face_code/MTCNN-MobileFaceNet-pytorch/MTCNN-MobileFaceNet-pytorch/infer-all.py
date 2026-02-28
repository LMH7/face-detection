import argparse
import functools
import os
import time
from tqdm import tqdm  # 添加进度条支持

import cv2
import numpy as np
import torch
from PIL import ImageDraw, ImageFont, Image

from detection.face_detect import MTCNN
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('input_path',               str,     'dataset/images',                     '输入图片路径或文件夹路径')
add_arg('output_path',              str,     'dataset/images_results3',                     '结果保存路径')
add_arg('face_db_path',             str,     'face_db_new',                 '人脸库路径')
add_arg('threshold',                float,   0.4,                           '判断相识度的阈值')
add_arg('mobilefacenet_model_path', str,     'save_model/mobilefacenet_old.pth', 'MobileFaceNet预测模型的路径')
add_arg('mtcnn_model_path',         str,     'save_model/mtcnn',            'MTCNN预测模型的路径')
args = parser.parse_args()
print_arguments(args)


class Predictor:
    def __init__(self, mtcnn_model_path, mobilefacenet_model_path, face_db_path, threshold=0.7):
        self.threshold = threshold
        self.mtcnn = MTCNN(model_path=mtcnn_model_path)
        self.device = torch.device("cuda")

        # 加载模型
        self.model = torch.jit.load(mobilefacenet_model_path)
        self.model.to(self.device)
        self.model.eval()

        self.faces_db = self.load_face_db(face_db_path)

    def load_face_db(self, face_db_path):
        faces_db = {}
        for path in os.listdir(face_db_path):
            if path.startswith('.') or os.path.isdir(os.path.join(face_db_path, path)):
                continue
            name = os.path.basename(path).split('.')[0]
            image_path = os.path.join(face_db_path, path)
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            imgs, _ = self.mtcnn.infer_image(img)
            if imgs is None or len(imgs) > 1:
                print('人脸库中的 %s 图片包含不是1张人脸，自动跳过该图片' % image_path)
                continue
            imgs = self.process(imgs)
            feature = self.infer(imgs[0])
            faces_db[name] = feature[0][0]
        return faces_db

    @staticmethod
    def process(imgs):
        imgs1 = []
        for img in imgs:
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 127.5
            imgs1.append(img)
        return imgs1

    def infer(self, imgs):
        assert len(imgs.shape) == 3 or len(imgs.shape) == 4
        if len(imgs.shape) == 3:
            imgs = imgs[np.newaxis, :]
        features = []
        for i in range(imgs.shape[0]):
            img = imgs[i][np.newaxis, :]
            img = torch.tensor(img, dtype=torch.float32, device=self.device)
            feature = self.model(img)
            feature = feature.detach().cpu().numpy()
            features.append(feature)
        return features

    def recognition(self, image_path):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        imgs, boxes = self.mtcnn.infer_image(img)
        if imgs is None:
            return None, None, None
        imgs = self.process(imgs)
        imgs = np.array(imgs, dtype='float32')
        features = self.infer(imgs)
        names = []
        probs = []
        for i in range(len(features)):
            feature = features[i][0]
            results_dict = {}
            for name in self.faces_db.keys():
                feature1 = self.faces_db[name]
                prob = np.dot(feature, feature1) / (np.linalg.norm(feature) * np.linalg.norm(feature1))
                results_dict[name] = prob
            results = sorted(results_dict.items(), key=lambda d: d[1], reverse=True)
            result = results[0]
            prob = float(result[1])
            probs.append(prob)
            if prob > self.threshold:
                name = result[0]
                names.append(name)
            else:
                names.append('unknow')
        return boxes, names, probs

    def draw_face(self, img, boxes_c, names, probs=None):
        if boxes_c is None:
            return img
            
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
    
        try:
            font = ImageFont.truetype("simhei.ttf", 20)
        except:
            font = ImageFont.load_default()
    
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            name = names[i]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        
            color = "green" if name != 'unknow' else "red"
            draw.rectangle(
                [corpbbox[0], corpbbox[1], corpbbox[2], corpbbox[3]],
                outline=color,
                width=2
            )
        
            if name != 'unknow' and probs is not None:
                text = f"{name} {probs[i]:.2f}"
            else:
                text = f"{name}"
        
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        
            draw.rectangle(
                [corpbbox[0], corpbbox[1] - text_height - 10,
                corpbbox[0] + text_width, corpbbox[1]],
                fill=color
            )
        
            draw.text(
                (corpbbox[0], corpbbox[1] - text_height - 5),
                text,
                font=font,
                fill="white"
            )
    
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def process_folder(predictor, input_path, output_path):
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    if os.path.isfile(input_path):
        image_paths = [input_path]
    else:
        image_paths = []
        for root, _, files in os.walk(input_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
    
    # 处理每张图片
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # 处理图片
            boxes, names, probs = predictor.recognition(image_path)
            
            # 读取原始图片
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            
            # 绘制结果
            if boxes is not None:
                img = predictor.draw_face(img, boxes, names, probs)
            
            # 保存结果
            rel_path = os.path.relpath(image_path, input_path)
            save_path = os.path.join(output_path, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img)
            
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {str(e)}")
            continue


if __name__ == '__main__':
    # 初始化预测器
    predictor = Predictor(args.mtcnn_model_path, 
                         args.mobilefacenet_model_path, 
                         args.face_db_path, 
                         threshold=args.threshold)
    
    # 处理输入路径
    start_time = time.time()
    process_folder(predictor, args.input_path, args.output_path)
    
    print(f'处理完成，总耗时: {time.time()-start_time:.2f}秒')
    print(f'结果已保存到: {args.output_path}')
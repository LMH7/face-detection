import argparse
import functools
import os
import time

import cv2
import numpy as np
import torch
from PIL import ImageDraw, ImageFont, Image

from detection.face_detect import MTCNN
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('image_path',               str,     'dataset/DJI_0002_2_000008.jpg',    '预测图片路径')
add_arg('face_db_path',             str,     'face_db_new',                          '人脸库路径')
add_arg('threshold',                float,   0.2,                                '判断相识度的阈值')
add_arg('mobilefacenet_model_path', str,     'save_model/mobilefacenet_old.pth',     'MobileFaceNet预测模型的路径')
add_arg('mtcnn_model_path',         str,     'save_model/mtcnn',                 'MTCNN预测模型的路径')
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

    # 预测图片
    def infer(self, imgs):
        assert len(imgs.shape) == 3 or len(imgs.shape) == 4
        if len(imgs.shape) == 3:
            imgs = imgs[np.newaxis, :]
        # TODO 不知为何不支持多张图片预测
        '''
        imgs = torch.tensor(imgs, dtype=torch.float32, device=self.device)
        features = self.model(img)
        features = features.detach().cpu().numpy()
        '''
        features = []
        for i in range(imgs.shape[0]):
            img = imgs[i][np.newaxis, :]
            img = torch.tensor(img, dtype=torch.float32, device=self.device)
            # 执行预测
            feature = self.model(img)
            feature = feature.detach().cpu().numpy()
            features.append(feature)
        return features

    def recognition(self, image_path):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        s = time.time()
        imgs, boxes = self.mtcnn.infer_image(img)
        print('人脸检测时间：%dms' % int((time.time() - s) * 1000))
        if imgs is None:
            return None, None
        imgs = self.process(imgs)
        imgs = np.array(imgs, dtype='float32')
        s = time.time()
        features = self.infer(imgs)
        print('人脸识别时间：%dms' % int((time.time() - s) * 1000))
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
            print('人脸对比结果：', results)
            result = results[0]
            prob = float(result[1])
            probs.append(prob)
            if prob > self.threshold:
                name = result[0]
                names.append(name)
            else:
                names.append('unknow')
        return boxes, names, probs  # 增加返回probs

    def add_text(self, img, text, left, top, color=(0, 0, 0), size=20):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('simfang.ttf', size)
        draw.text((left, top), text, color, font=font)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 画出人脸框和关键点
    def draw_face(self, image_path, boxes_c, names, probs=None):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        if boxes_c is not None:
            # 转换为PIL图像
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
            
                # 使用PIL绘制人脸框
                color = "green" if name != 'unknow' else "red"
                draw.rectangle(
                    [corpbbox[0], corpbbox[1], corpbbox[2], corpbbox[3]],
                    outline=color,
                    width=2
                )
            
                # 准备文本
                if name != 'unknow' and probs is not None:
                    text = f"{name} {probs[i]:.2f}"
                else:
                    text = f"{name}"
            
                # 获取文本尺寸
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            
                # 画文字背景框
                draw.rectangle(
                    [corpbbox[0], corpbbox[1] - text_height - 10,
                    corpbbox[0] + text_width, corpbbox[1]],
                    fill=color
                )
            
                # 写文字
                draw.text(
                    (corpbbox[0], corpbbox[1] - text_height - 5),
                    text,
                    font=font,
                    fill="white"
                )
        
            # 转换回OpenCV格式
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
        # 保存结果图片
        output_path = os.path.splitext(image_path)[0] + "_result.jpg"
        cv2.imwrite(output_path, img)
        print(f"结果已保存到: {output_path}")
    
        # 显示结果（可缩放窗口）
        cv2.namedWindow("Face Recognition Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Face Recognition Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    predictor = Predictor(args.mtcnn_model_path, 
                         args.mobilefacenet_model_path, 
                         args.face_db_path, 
                         threshold=args.threshold)
    start = time.time()
    boxes, names, probs = predictor.recognition(args.image_path)  # 接收probs
    print('预测的人脸位置：', boxes.astype(np.int_).tolist())
    print('识别的人脸名称：', names)
    print('识别置信度：', probs)
    print('总识别时间：%dms' % int((time.time() - start) * 1000))
    predictor.draw_face(args.image_path, boxes, names, probs)  # 传入probs
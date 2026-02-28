import os
import random
from collections import defaultdict

testset_dir = "dataset/test"  # 你的数据集路径
output_file = "dataset/test/test.txt"      # 输出文件
base_pairs_per_person = 3           # 每人最少正样本对数
scale_factor = 1                 # 数量缩放系数
valid_extensions = {'.jpg', '.jpeg', '.png'}  # 允许的图片格式

# 获取所有人物及其图像（过滤空文件夹和非图片文件）
people_images = defaultdict(list)
for person in os.listdir(testset_dir):
    person_dir = os.path.join(testset_dir, person)
    if not os.path.isdir(person_dir):
        continue
        
    # 只收集有效图片文件
    images = [
        img for img in os.listdir(person_dir) 
        if os.path.splitext(img)[1].lower() in valid_extensions
    ]
    
    if len(images) >= 2:  # 只保留至少有两张图片的人物
        people_images[person] = images

# 检查是否有有效数据
if not people_images:
    raise ValueError("没有找到有效的图片数据！请检查数据集路径和文件格式。")

# 生成正样本对
pairs = []
used_pairs = set()

for person, images in people_images.items():
    num_pos_pairs = base_pairs_per_person + int(len(images) * scale_factor)
    num_pos_pairs = min(num_pos_pairs, len(images))
    
    for _ in range(num_pos_pairs):
        img1, img2 = random.sample(images, 2)
        pair_key = f"{testset_dir}/{person}/{img1} {testset_dir}/{person}/{img2}"
        if pair_key not in used_pairs:
            pairs.append(f"{pair_key} 1")
            used_pairs.add(pair_key)

# 生成负样本对
people_list = list(people_images.keys())
num_neg_pairs = len(pairs)

for _ in range(num_neg_pairs):
    # 确保选择两个不同且非空的人物
    person1, person2 = random.sample(people_list, 2)
    img1 = random.choice(people_images[person1])
    img2 = random.choice(people_images[person2])
    pair_key = f"{testset_dir}/{person1}/{img1} {testset_dir}/{person2}/{img2}"
    if pair_key not in used_pairs:
        pairs.append(f"{pair_key} 0")
        used_pairs.add(pair_key)

# 写入文件
with open(output_file, "w") as f:
    f.write("\n".join(pairs))
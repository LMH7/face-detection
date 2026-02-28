import os
import re

def rename_files(folder_path):
    """
    将指定文件夹中所有文件名中的括号()替换为-，并去除空格
    例如：DJI_0001 (2)_000009_face_0.jpg → DJI_0001-2_000009_face_0.jpg
    """
    for filename in os.listdir(folder_path):
        # 替换括号和空格
        new_name = re.sub(r'[()]', '_', filename)  # 替换()为-
        new_name = new_name.replace(' ', '')       # 去除空格

        if new_name != filename:
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            # 重命名文件
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_name}")

# 使用示例
folder_path = "images"  # 替换为你的图片文件夹路径
rename_files(folder_path)
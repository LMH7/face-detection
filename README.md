# 高空无人机视角人脸识别项目 (High-Altitude Drone Face Recognition)

## 📌 项目简介
本项目基于原有的 MobileFaceNet 和 MTCNN 进行深度二次开发。针对高空无人机拍摄视角下人脸尺寸小、角度偏移大等痛点，使用了特定的**高空无人机人脸数据集**进行迁移学习和优化。

## 🛠 主要改进
- **视角优化**：针对无人机俯视角度进行了模型微调。
- **数据集更新**：替换了原有的通用人脸数据集，改用高空无人机专项数据集。
- **环境适配**：项目已在 AutoDL 云服务器环境配置完成，支持 PyTorch/PaddlePaddle 框架。

## 📂 目录结构
- `face_code/`: 基于 PyTorch 的实现，包含高空视角优化逻辑。
- `PaddlePaddle-MobileFaceNets-master/`: 基于 PaddlePaddle 的实验版本。
-  里面添加了关于每个人识别准确率分析，imges里面有图片展示

## 🚀 快速开始
1. 准备数据集：请将高空人脸数据集放置在各项目文件夹的 `dataset/` 目录下。
2. 运行训练脚本：`python train.py`

## ❤️ 致谢与参考 (Acknowledgements)

本项目在开发过程中深度参考了以下优秀开源项目，特此致谢：

- **项目名称**：[PaddlePaddle-MobileFaceNets](https://github.com/yeyupiaoling/PaddlePaddle-MobileFaceNets)
- **原作者**：[yeyupiaoling](https://github.com/yeyupiaoling)
- **项目简介**：该项目提供了基于 PaddlePaddle 实现的 MobileFaceNets 人脸识别模型，并结合 MTCNN 进行人脸检测，为本项目的“高空视角优化”提供了坚实的基础。
---
*注：本项目代码库仅作个人实验备份使用。*

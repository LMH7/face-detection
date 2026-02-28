import os
import numpy as np
import paddle
from collections import defaultdict

from utils.arcmargin import ArcNet
from utils.mobilefacenet import MobileFaceNet
from utils.utils import get_lfw_list, get_features, get_feature_dict, test_performance, cosin_metric


# ===============================
# 从路径中提取person ID
# ===============================
def extract_person_id(img_path):
    """从图片路径中提取person ID，例如：dataset/test/person01/xxx.jpg -> person01"""
    parts = img_path.split(os.sep)
    for part in parts:
        if part.startswith('person'):
            return part
    return None


# ===============================
# 读取测试列表文件
# ===============================
def read_test_list(test_list_path):
    """读取测试列表文件，格式：图片1路径 图片2路径 标签(1/0)"""
    pairs = []
    with open(test_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                img1_path = parts[0]
                img2_path = parts[1]
                label = int(parts[2])
                pairs.append((img1_path, img2_path, label))
    return pairs


# ===============================
# 按person统计准确率
# ===============================
@paddle.no_grad()
def test_per_person(model, test_list_path, threshold, fe_dict):
    """
    按person统计准确率
    Args:
        model: 训练好的模型（已设置为eval模式）
        test_list_path: 测试列表文件路径
        threshold: 相似度阈值
        fe_dict: 特征字典
    """
    # 读取测试列表
    pairs = read_test_list(test_list_path)
    print(f"[INFO] 读取到 {len(pairs)} 对测试数据")

    # 按person分组统计
    person_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'true_positive': 0, 'false_positive': 0,
                                        'true_negative': 0, 'false_negative': 0})

    # 总体统计
    total_correct = 0
    total_samples = 0
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    # 遍历所有测试对
    for img1_path, img2_path, true_label in pairs:
        # 获取特征
        if img1_path not in fe_dict or img2_path not in fe_dict:
            continue

        feature1 = fe_dict[img1_path]
        feature2 = fe_dict[img2_path]

        # 计算相似度（使用与utils.py相同的cosin_metric函数）
        similarity = cosin_metric(feature1, feature2)

        # 预测标签（相似度大于阈值认为是同一个人）
        pred_label = 1 if similarity > threshold else 0

        # 判断预测是否正确
        is_correct = (pred_label == true_label)

        # 总体统计
        total_samples += 1
        if is_correct:
            total_correct += 1

        if true_label == 1 and pred_label == 1:
            total_tp += 1
        elif true_label == 1 and pred_label == 0:
            total_fn += 1
        elif true_label == 0 and pred_label == 1:
            total_fp += 1
        elif true_label == 0 and pred_label == 0:
            total_tn += 1

        # 从路径中提取person ID
        person1_id = extract_person_id(img1_path)
        person2_id = extract_person_id(img2_path)

        # 统计每个person的结果
        # 如果两个图片属于同一个person，这个测试对属于该person（标签应该为1）
        if person1_id and person1_id == person2_id:
            person_stats[person1_id]['total'] += 1
            if is_correct:
                person_stats[person1_id]['correct'] += 1
            # 统计TP, FP, TN, FN
            if true_label == 1 and pred_label == 1:
                person_stats[person1_id]['true_positive'] += 1
            elif true_label == 1 and pred_label == 0:
                person_stats[person1_id]['false_negative'] += 1
            elif true_label == 0 and pred_label == 1:
                person_stats[person1_id]['false_positive'] += 1
            elif true_label == 0 and pred_label == 0:
                person_stats[person1_id]['true_negative'] += 1
        # 如果两个图片属于不同person，这个测试对同时属于两个person（标签应该为0）
        elif person1_id and person2_id and person1_id != person2_id:
            # 对于person1
            person_stats[person1_id]['total'] += 1
            if is_correct:
                person_stats[person1_id]['correct'] += 1
            if true_label == 0 and pred_label == 0:
                person_stats[person1_id]['true_negative'] += 1
            elif true_label == 0 and pred_label == 1:
                person_stats[person1_id]['false_positive'] += 1
            elif true_label == 1 and pred_label == 1:
                person_stats[person1_id]['true_positive'] += 1
            elif true_label == 1 and pred_label == 0:
                person_stats[person1_id]['false_negative'] += 1

            # 对于person2
            person_stats[person2_id]['total'] += 1
            if is_correct:
                person_stats[person2_id]['correct'] += 1
            if true_label == 0 and pred_label == 0:
                person_stats[person2_id]['true_negative'] += 1
            elif true_label == 0 and pred_label == 1:
                person_stats[person2_id]['false_positive'] += 1
            elif true_label == 1 and pred_label == 1:
                person_stats[person2_id]['true_positive'] += 1
            elif true_label == 1 and pred_label == 0:
                person_stats[person2_id]['false_negative'] += 1

    # 计算每个person的准确率、精确率、召回率、F1
    results = []

    # 按person ID排序
    sorted_persons = sorted(person_stats.keys())

    for person_id in sorted_persons:
        stats = person_stats[person_id]
        if stats['total'] == 0:
            continue

        accuracy = (stats['correct'] / stats['total']) * 100

        # 计算精确率、召回率、F1
        tp = stats['true_positive']
        fp = stats['false_positive']
        tn = stats['true_negative']
        fn = stats['false_negative']

        precision = (tp / (tp + fp + 1e-8)) * 100
        recall = (tp / (tp + fn + 1e-8)) * 100
        f1 = (2 * precision * recall / (precision + recall + 1e-8))

        results.append([person_id, precision, recall, f1, accuracy])

    # 计算总体指标
    if total_samples > 0:
        total_accuracy = (total_correct / total_samples) * 100
        total_precision = (total_tp / (total_tp + total_fp + 1e-8)) * 100
        total_recall = (total_tp / (total_tp + total_fn + 1e-8)) * 100
        total_f1 = (2 * total_precision * total_recall / (total_precision + total_recall + 1e-8))
    else:
        total_accuracy = total_precision = total_recall = total_f1 = 0.0

    # 添加总体统计
    results.append(["Total", total_precision, total_recall, total_f1, total_accuracy])

    return results


# ===============================
# 主程序入口
# ===============================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='按person统计准确率')
    parser.add_argument('--test_list_path', type=str, default='dataset/test/test.txt',
                        help='测试列表文件路径')
    parser.add_argument('--model_path', type=str, default='models/epoch_29',
                        help='模型路径')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批量大小')
    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.test_list_path):
        raise FileNotFoundError(f"测试列表文件不存在: {args.test_list_path}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型路径不存在: {args.model_path}")

    # 初始化模型（完全模仿train.py的方式）
    model = MobileFaceNet()

    # 加载模型参数
    model_state = paddle.load(os.path.join(args.model_path, 'model.pdparams'))
    model.set_state_dict(model_state)
    print(f"[INFO] 成功加载模型: {args.model_path}")

    # 设置为eval模式（完全模仿train.py的test函数）
    model.eval()

    # 获取测试数据并提取特征（完全模仿train.py的test函数）
    print("[INFO] 开始提取特征...")
    img_paths = get_lfw_list(args.test_list_path)
    features = get_features(model, img_paths, batch_size=args.batch_size)
    fe_dict = get_feature_dict(img_paths, features)
    print("[INFO] 特征提取完成")

    # 使用test_performance函数获取最佳阈值和总体准确率（完全模仿train.py）
    print("[INFO] 使用test_performance函数寻找最佳阈值...")
    overall_accuracy, best_threshold = test_performance(fe_dict, args.test_list_path)
    print(f"[INFO] 总体准确率: {overall_accuracy * 100:.2f}%")
    print(f"[INFO] 最佳阈值: {best_threshold:.6f}")

    # 验证：如果准确率低于80%，给出警告
    if overall_accuracy < 0.80:
        print(f"\n[WARN] 总体准确率 {overall_accuracy * 100:.2f}% 低于预期（应该接近90%）")
        print("[WARN] 请检查：")
        print("  1. 模型文件路径是否正确")
        print("  2. 模型是否训练完成（epoch_29）")
        print("  3. 测试数据文件是否正确")
        print("  4. 运行 train.py 的测试函数，确认实际准确率")

    # 使用最佳阈值进行按person统计
    print(f"\n[INFO] 开始按person统计准确率（使用阈值: {best_threshold:.6f}）...")
    per_person_result = test_per_person(model, args.test_list_path, best_threshold, fe_dict)

    # 打印结果表格
    print("\n表4-3 各类别 Precision / Recall / F1 / Accuracy 结果：\n")
    print("{:<12}{:<12}{:<12}{:<12}{:<12}".format("类别", "Precision", "Recall", "F1", "Accuracy"))
    print("-" * 60)
    for r in per_person_result:
        if len(r) == 5:
            print("{:<12}{:<12.2f}{:<12.2f}{:<12.2f}{:<12.2f}".format(*r))
        else:
            print("{:<12}{:<12.2f}{:<12.2f}{:<12.2f}".format(*r))

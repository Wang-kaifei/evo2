import json
import numpy as np
from pathlib import Path

def analyze_training_data(file_path):
    """分析训练数据中correct_scale字段中-1的占比
    
    Args:
        file_path: 训练数据文件路径
    """
    # 读取训练数据
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 统计信息
    total_positions = 0
    minus_one_positions = 0
    
    # 遍历每个样本
    for sample in data:
        correct_scales = sample['correct_scale']
        total_positions += len(correct_scales)
        minus_one_positions += correct_scales.count(-1)
    
    # 计算占比
    minus_one_ratio = minus_one_positions / total_positions * 100
    
    # 打印统计信息
    print(f"\n训练数据分析结果:")
    print(f"总样本数: {len(data)}")
    print(f"总位置数: {total_positions}")
    print(f"-1位置数: {minus_one_positions}")
    print(f"-1占比: {minus_one_ratio:.2f}%")
    
    # 分析每个scale的分布
    scale_counts = {}
    for sample in data:
        for scale in sample['correct_scale']:
            if scale != -1:  # 只统计非-1的值
                scale_counts[scale] = scale_counts.get(scale, 0) + 1
    
    print("\n各scale的分布:")
    for scale in sorted(scale_counts.keys()):
        count = scale_counts[scale]
        ratio = count / (total_positions - minus_one_positions) * 100
        print(f"Scale {scale}: {count} ({ratio:.2f}%)")

if __name__ == "__main__":
    file_path = "/root/EVO/evo2/scale_demo/data/training_data.json"
    analyze_training_data(file_path) 
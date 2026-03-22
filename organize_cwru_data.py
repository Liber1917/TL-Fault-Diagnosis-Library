#!/usr/bin/env python3
"""
CWRU 数据集整理脚本
将 data.zip 中的数据整理到 datasets/CWRU/ 目录下，按工况条件分类

工况条件对应关系：
- condition_0: 0 HP, 1797 RPM
- condition_1: 1 HP, 1772 RPM
- condition_2: 2 HP, 1750 RPM
- condition_3: 3 HP, 1730 RPM

故障类型对应关系：
- 内圈故障 (105,106,107,108,169,170,171,172,209,210,211,212) -> inner_07, inner_14, inner_21
- 滚动体故障 (118,119,120,121,185,186,187,188,222,223,224,225) -> ball_07, ball_14, ball_21
- 外圈故障 (130-133,197-200,234-237) -> outer_07, outer_14, outer_21 (6:00 位置)
- 正常数据 (97,98,99,100) -> normal
"""

import os
import zipfile
import shutil
from collections import defaultdict

# 文件编号到工况和故障类型的映射
# 格式：文件编号 -> (condition, fault_type)
# fault_size: 07=0.007", 14=0.014", 21=0.021"

FILE_MAPPING = {
    # 正常数据 - 单独处理，不放在故障类别中
    # 97: ('condition_0', 'normal'),
    # 98: ('condition_1', 'normal'),
    # 99: ('condition_2', 'normal'),
    # 100: ('condition_3', 'normal'),
    
    # 内圈故障 0.007"
    105: ('condition_0', 'inner_07'),
    106: ('condition_1', 'inner_07'),
    107: ('condition_2', 'inner_07'),
    108: ('condition_3', 'inner_07'),
    
    # 滚动体故障 0.007"
    118: ('condition_0', 'ball_07'),
    119: ('condition_1', 'ball_07'),
    120: ('condition_2', 'ball_07'),
    121: ('condition_3', 'ball_07'),
    
    # 外圈故障 0.007" (6:00 位置 - 负载区中心)
    130: ('condition_0', 'outer_07'),
    131: ('condition_1', 'outer_07'),
    132: ('condition_2', 'outer_07'),
    133: ('condition_3', 'outer_07'),
    
    # 内圈故障 0.014"
    169: ('condition_0', 'inner_14'),
    170: ('condition_1', 'inner_14'),
    171: ('condition_2', 'inner_14'),
    172: ('condition_3', 'inner_14'),
    
    # 滚动体故障 0.014"
    185: ('condition_0', 'ball_14'),
    186: ('condition_1', 'ball_14'),
    187: ('condition_2', 'ball_14'),
    188: ('condition_3', 'ball_14'),
    
    # 外圈故障 0.014" (6:00 位置)
    197: ('condition_0', 'outer_14'),
    198: ('condition_1', 'outer_14'),
    199: ('condition_2', 'outer_14'),
    200: ('condition_3', 'outer_14'),
    
    # 内圈故障 0.021"
    209: ('condition_0', 'inner_21'),
    210: ('condition_1', 'inner_21'),
    211: ('condition_2', 'inner_21'),
    212: ('condition_3', 'inner_21'),
    
    # 滚动体故障 0.021"
    222: ('condition_0', 'ball_21'),
    223: ('condition_1', 'ball_21'),
    224: ('condition_2', 'ball_21'),
    225: ('condition_3', 'ball_21'),
    
    # 外圈故障 0.021" (6:00 位置)
    234: ('condition_0', 'outer_21'),
    235: ('condition_1', 'outer_21'),
    236: ('condition_2', 'outer_21'),
    237: ('condition_3', 'outer_21'),
}

# 外圈故障其他位置（可选，如果需要更细粒度的分类）
# 3:00 位置（正交）
OUTER_3OCLOCK = {
    144: ('condition_0', 'outer_07'),
    145: ('condition_1', 'outer_07'),
    146: ('condition_2', 'outer_07'),
    147: ('condition_3', 'outer_07'),
}

# 12:00 位置（相对）
OUTER_12OCLOCK = {
    156: ('condition_0', 'outer_07'),
    158: ('condition_1', 'outer_07'),
    159: ('condition_2', 'outer_07'),
    160: ('condition_3', 'outer_07'),
}


def organize_cwru_data(zip_path, output_dir):
    """
    整理 CWRU 数据集
    
    Args:
        zip_path: data.zip 文件路径
        output_dir: 输出目录（datasets/CWRU/）
    """
    print(f"正在处理：{zip_path}")
    print(f"输出目录：{output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开 zip 文件
    with zipfile.ZipFile(zip_path, 'r') as z:
        # 获取所有 .mat 文件
        mat_files = [n for n in z.namelist() if n.endswith('.mat')]
        print(f"找到 {len(mat_files)} 个 .mat 文件")
        
        # 按 condition 和 fault_type 分组
        grouped = defaultdict(list)
        
        for file_path in mat_files:
            # 提取文件编号
            filename = os.path.basename(file_path)
            try:
                file_num = int(filename.split('.')[0])
            except ValueError:
                print(f"跳过无法解析的文件：{file_path}")
                continue
            
            # 查找对应的 condition 和 fault_type
            if file_num in FILE_MAPPING:
                condition, fault_type = FILE_MAPPING[file_num]
                grouped[(condition, fault_type)].append(file_path)
        
        # 复制到目标目录
        for (condition, fault_type), files in grouped.items():
            target_dir = os.path.join(output_dir, condition, fault_type)
            os.makedirs(target_dir, exist_ok=True)
            
            for file_path in files:
                # 从 zip 中提取文件
                content = z.read(file_path)
                filename = os.path.basename(file_path)
                target_path = os.path.join(target_dir, filename)
                
                with open(target_path, 'wb') as f:
                    f.write(content)
                
                print(f"  {file_path} -> {target_path}")
    
    # 统计结果
    print("\n整理完成！")
    print("=" * 50)
    for condition in ['condition_0', 'condition_1', 'condition_2', 'condition_3']:
        condition_dir = os.path.join(output_dir, condition)
        if os.path.exists(condition_dir):
            fault_types = os.listdir(condition_dir)
            total_files = sum(
                len(os.listdir(os.path.join(condition_dir, ft)))
                for ft in fault_types
            )
            print(f"{condition}: {len(fault_types)} 类故障，共 {total_files} 个文件")


if __name__ == '__main__':
    import sys
    
    # 默认路径
    zip_path = '/home/data.zip'
    output_dir = '/home/TL-Fault-Diagnosis-Library/datasets/CWRU'
    
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    organize_cwru_data(zip_path, output_dir)

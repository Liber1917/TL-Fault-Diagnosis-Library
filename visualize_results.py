#!/usr/bin/env python3
"""
Many-to-One Transfer 实验结果可视化
- 训练曲线（Loss 和 Accuracy）
- 从日志文件提取数据并绘图
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def parse_log(log_path):
    """从日志文件解析训练数据"""
    epochs = []
    loss_classifier = []
    loss_mmd = []
    loss_l1 = []
    train_acc = []
    val_acc = []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 提取每个 epoch 的数据
    epoch_pattern = r'-----Epoch (\d+)/(\d+)-----'
    loss_pattern = r'Train-Loss Source Classifier: ([\d.]+)'
    mmd_pattern = r'Train-Loss MMD: ([\d.]+)'
    l1_pattern = r'Train-Loss L1: ([\d.]+)'
    train_acc_pattern = r'Train-Acc Source Data: ([\d.]+)'
    val_acc_pattern = r'Val-acc: ([\d.]+)'
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'Train-Loss Source Classifier:' in line:
            match = re.search(loss_pattern, line)
            if match:
                loss_classifier.append(float(match.group(1)))
        
        if 'Train-Loss MMD:' in line:
            match = re.search(mmd_pattern, line)
            if match:
                loss_mmd.append(float(match.group(1)))
        
        if 'Train-Loss L1:' in line:
            match = re.search(l1_pattern, line)
            if match:
                loss_l1.append(float(match.group(1)))
        
        if 'Train-Acc Source Data:' in line:
            match = re.search(train_acc_pattern, line)
            if match:
                train_acc.append(float(match.group(1)))
        
        if 'Val-acc:' in line:
            match = re.search(val_acc_pattern, line)
            if match:
                val_acc.append(float(match.group(1)))
    
    epochs = list(range(1, len(loss_classifier) + 1))
    
    return {
        'epochs': epochs,
        'loss_classifier': loss_classifier,
        'loss_mmd': loss_mmd,
        'loss_l1': loss_l1,
        'train_acc': train_acc,
        'val_acc': val_acc
    }


def plot_training_curves(data, save_path=None):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = data['epochs']
    
    # Loss 曲线
    ax1 = axes[0]
    ax1.plot(epochs, data['loss_classifier'], 'b-o', label='Classifier Loss', linewidth=2, markersize=8)
    ax1.plot(epochs, data['loss_mmd'], 'r-s', label='MMD Loss', linewidth=2, markersize=8)
    ax1.plot(epochs, data['loss_l1'], 'g-^', label='L1 Loss', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Losses', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Accuracy 曲线
    ax2 = axes[1]
    ax2.plot(epochs, data['train_acc'], 'b-o', label='Source Train Acc', linewidth=2, markersize=8)
    ax2.plot(epochs, data['val_acc'], 'r-s', label='Target Val Acc', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    ax2.set_ylim([0, 1.05])
    
    # 在最终准确率处添加标注
    final_val_acc = data['val_acc'][-1]
    ax2.annotate(f'Final: {final_val_acc*100:.1f}%', 
                 xy=(epochs[-1], final_val_acc),
                 xytext=(epochs[-1]-0.5, final_val_acc-0.1),
                 fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.close()
    return fig


def plot_comparison(results_dict, save_path=None):
    """绘制多个实验的对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, data) in enumerate(results_dict.items()):
        epochs = data['epochs']
        ax.plot(epochs, data['val_acc'], 
                color=colors[i % len(colors)], 
                marker='o', 
                linewidth=2, 
                markersize=8,
                label=name)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Many-to-One Transfer: Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.close()
    return fig


def generate_summary(data, experiment_name):
    """生成实验总结"""
    print("\n" + "="*60)
    print(f"  {experiment_name} - 实验结果总结")
    print("="*60)
    print(f"  训练 Epochs: {len(data['epochs'])}")
    print(f"  最终源域训练准确率: {data['train_acc'][-1]*100:.2f}%")
    print(f"  最终目标域验证准确率: {data['val_acc'][-1]*100:.2f}%")
    print(f"  最佳验证准确率: {max(data['val_acc'])*100:.2f}% (Epoch {data['val_acc'].index(max(data['val_acc']))+1})")
    print("="*60 + "\n")


if __name__ == '__main__':
    import sys
    
    # 默认日志文件
    log_dir = './ckpt/MFSAN/multi_source'
    log_file = '[CWRU_0_CWRU_1]To[CWRU_2]_0322-161840.log'
    log_path = os.path.join(log_dir, log_file)
    
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    
    print(f"Parsing log: {log_path}")
    
    # 解析日志
    data = parse_log(log_path)
    
    # 生成总结
    experiment_name = "Many-to-One Transfer (CWRU_0,CWRU_1 → CWRU_2)"
    generate_summary(data, experiment_name)
    
    # 绘制训练曲线
    output_dir = os.path.dirname(log_path)
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plot_training_curves(data, plot_path)
    
    print("可视化完成！")
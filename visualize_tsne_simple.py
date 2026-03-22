#!/usr/bin/env python3
"""
t-SNE 特征可视化 - 简化版
直接使用训练好的模型提取特征并可视化
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from opt import parse_args

sys.path.extend(['./models', './data_loader'])
import utils
import modules
from data_loader import load_methods
from data_loader import conditional_load, data_utils


def get_fault_label_mapping(args):
    """获取故障标签映射"""
    args.faults, args.num_classes = [], []
    for name in args.source_name + [args.target]:
        dataset, condition, selected_list = utils.get_info_from_name(name)
        if condition is not None:
            data_root = os.path.join(args.data_dir, dataset)
            faults = np.array(sorted(os.listdir(os.path.join(data_root, 'condition_%d' % condition))))
        else:
            data_root = os.path.join(args.data_dir, dataset)
            faults = np.array(sorted(os.listdir(data_root)))
        if selected_list:
            faults = faults[selected_list]
        num_classes = len(faults)
        args.faults.append(faults)
        args.num_classes.append(num_classes)
    
    # 获取所有故障类别
    all_faults = set()
    for faults in args.faults:
        for item in faults:
            all_faults.add(item)
    args.fault_label = {}
    for i, fault in enumerate(sorted(all_faults)):
        args.fault_label[fault] = i
    
    # 获取 label_sets
    args.label_sets = list()
    for faults in args.faults:
        args.label_sets.append([args.fault_label[item] for item in faults])
    
    return args


def load_mfsan_model(model_path, args):
    """加载 MFSAN 模型"""
    # 初始化模型组件
    if args.backbone == 'CNN':
        G = modules.MSCNN(in_channel=1).to('cpu')
    else:
        G = modules.ResNet(in_channel=1, layers=[2, 2, 2, 2]).to('cpu')
    
    num_classes = len(args.label_sets[0])  # 使用第一个源域的类别数
    Fs = torch.nn.ModuleList([
        modules.MLP(input_size=G.out_dim, dropout=args.dropout, num_layer=2, output_layer=False)
        for _ in range(len(args.source_name))
    ]).to('cpu')
    Cs = torch.nn.ModuleList([
        modules.MLP(input_size=Fs[i].feature_dim, output_size=num_classes,
                   num_layer=1, last=None) for i in range(len(args.source_name))
    ]).to('cpu')
    
    # 加载权重
    ckpt = torch.load(model_path, map_location='cpu')
    G.load_state_dict(ckpt['G'])
    Fs.load_state_dict(ckpt['Fs'])
    Cs.load_state_dict(ckpt['Cs'])
    
    G.eval()
    Fs.eval()
    Cs.eval()
    
    return G, Fs, Cs


def extract_features(G, Fs, Cs, dataloader, device='cpu'):
    """提取特征"""
    features = []
    labels = []
    
    with torch.no_grad():
        for data, actual_labels in dataloader:
            data = data.to(device)
            feat = G(data)
            features.append(feat.cpu().numpy())
            labels.append(actual_labels.numpy())
    
    return np.vstack(features), np.concatenate(labels)


def prepare_dataloaders(args):
    """准备数据加载器"""
    import importlib
    from torch.utils.data import DataLoader
    
    # 初始化数据集
    datasets = {}
    
    # 源域数据
    for i, source in enumerate(args.source_name):
        dataset, condition, _ = utils.get_info_from_name(source)
        Dataset = importlib.import_module("data_loader.conditional_load").dataset
        datasets[source] = Dataset(args, dataset, i, condition=condition).data_preprare(is_src=True)
    
    # 目标域数据
    dataset, condition, _ = utils.get_info_from_name(args.target)
    Dataset = importlib.import_module("data_loader.conditional_load").dataset
    datasets['train'], datasets['val'] = Dataset(args, dataset, -1, condition=condition).data_preprare(is_src=False)
    
    # 创建 DataLoader
    dataloaders = {}
    for key, ds in datasets.items():
        dataloaders[key] = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    
    return dataloaders


def plot_tsne(features, labels, domains, fault_names, save_path=None):
    """绘制 t-SNE 可视化"""
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 按故障类别着色
    ax1 = axes[0]
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=fault_names.get(label, f'Class {label}'),
                   alpha=0.6, s=30)
    
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.set_title('Feature Distribution by Fault Type', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 按域着色
    ax2 = axes[1]
    domain_colors = ['#1f77b4', '#ff7f0e']
    domain_names = ['Source', 'Target']
    
    for i, domain in enumerate([0, 1]):
        mask = domains == domain
        ax2.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=domain_colors[i], label=domain_names[i],
                   alpha=0.6, s=30)
    
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.set_title('Feature Distribution by Domain', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"t-SNE plot saved to: {save_path}")
    
    plt.close()


if __name__ == '__main__':
    
    # 参数
    model_path = './ckpt/MFSAN/multi_source/[CWRU_0_CWRU_1]To[CWRU_2]_0322-161840.pth'
    
    # 解析参数
    args = parse_args()
    args.model_name = 'MFSAN'
    args.source = 'CWRU_0,CWRU_1'
    args.target = 'CWRU_2'
    args.train_mode = 'multi_source'
    args.cuda_device = ''
    args.data_dir = './datasets'
    args.backbone = 'CNN'
    args.dropout = 0.0
    args.random_state = 10
    
    # 解析 source_name
    args.source_name = [x.strip() for x in list(args.source.split(','))]
    if '' in args.source_name:
        args.source_name.remove('')
    
    # 获取故障标签映射
    args = get_fault_label_mapping(args)
    
    # 故障类别名称映射
    fault_names = {
        0: 'ball_07', 1: 'ball_14', 2: 'ball_21',
        3: 'inner_07', 4: 'inner_14', 5: 'inner_21',
        6: 'outer_07', 7: 'outer_14', 8: 'outer_21'
    }
    
    print("Loading model...")
    G, Fs, Cs = load_mfsan_model(model_path, args)
    
    print("Preparing dataloaders...")
    dataloaders = prepare_dataloaders(args)
    
    # 提取源域特征（合并两个源域）
    print("Extracting source features...")
    src_features_list = []
    src_labels_list = []
    for source in args.source_name:
        feats, labels = extract_features(G, Fs, Cs, dataloaders[source])
        src_features_list.append(feats)
        src_labels_list.append(labels)
    
    src_features = np.vstack(src_features_list)
    src_labels = np.concatenate(src_labels_list)
    src_domains = np.zeros(len(src_labels))
    
    # 提取目标域特征
    print("Extracting target features...")
    tgt_features, tgt_labels = extract_features(G, Fs, Cs, dataloaders['val'])
    tgt_domains = np.ones(len(tgt_labels))
    
    # 合并
    all_features = np.vstack([src_features, tgt_features])
    all_labels = np.concatenate([src_labels, tgt_labels])
    all_domains = np.concatenate([src_domains, tgt_domains])
    
    print(f"Total samples: {len(all_labels)}")
    print(f"Source samples: {np.sum(all_domains == 0)}")
    print(f"Target samples: {np.sum(all_domains == 1)}")
    
    # 绘制 t-SNE
    output_path = './ckpt/MFSAN/multi_source/tsne_visualization.png'
    plot_tsne(all_features, all_labels, all_domains, fault_names, output_path)
    
    print("Done!")
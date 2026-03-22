# Many-to-One Transfer 复现指南

本指南帮助同学复现 TL-Fault-Diagnosis-Library 中的 Many-to-One Transfer 实验。

## 环境要求

- Python >= 3.8
- PyTorch >= 1.10
- NumPy >= 1.21.2
- Pandas >= 1.5.3
- tqdm >= 4.46.1
- SciPy >= 1.10
- matplotlib >= 3.7.0
- scikit-learn >= 1.3.0

### 安装依赖

```bash
pip3 install torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip3 install numpy pandas tqdm scipy matplotlib scikit-learn
```

## 数据集准备

### 1. 下载原始数据

从以下地址下载 CWRU 轴承数据集：
- 官网：https://engineering.case.edu/bearingdatacenter
- 或使用 `/home/data.zip`（如果已提供）

### 2. 运行数据整理脚本

```bash
cd /home/TL-Fault-Diagnosis-Library
python3 organize_cwru_data.py
```

这将把数据整理为以下结构：

```
datasets/CWRU/
├── condition_0/    # 0 HP, 1797 RPM
│   ├── ball_07/
│   ├── ball_14/
│   ├── ball_21/
│   ├── inner_07/
│   ├── inner_14/
│   ├── inner_21/
│   ├── outer_07/
│   ├── outer_14/
│   └── outer_21/
├── condition_1/    # 1 HP, 1772 RPM
├── condition_2/    # 2 HP, 1750 RPM
└── condition_3/    # 3 HP, 1730 RPM
```

**注意**：每个 condition 文件夹包含 9 个故障类别（不含 normal）：
- 滚动体故障：ball_07, ball_14, ball_21
- 内圈故障：inner_07, inner_14, inner_21
- 外圈故障：outer_07, outer_14, outer_21

## 运行实验

### Many-to-One Transfer（域内迁移）

```bash
cd /home/TL-Fault-Diagnosis-Library

python3 train.py \
    --model_name MFSAN \
    --source CWRU_0,CWRU_1 \
    --target CWRU_2 \
    --train_mode multi_source \
    --cuda_device 0 \
    --max_epoch 30
```

参数说明：
- `--model_name`: 模型名称（MFSAN, MSSA, ADACL）
- `--source`: 源域，多个用逗号分隔
- `--target`: 目标域
- `--train_mode`: 训练模式，multi_source 表示多源迁移
- `--cuda_device`: GPU 设备号，空字符串表示使用 CPU
- `--max_epoch`: 训练轮数

### 跨数据集 Many-to-One Transfer

```bash
python3 train.py \
    --model_name MFSAN \
    --source CWRU,PU \
    --target MFPT \
    --train_mode multi_source \
    --cuda_device 0
```

## 实验结果

训练完成后，结果保存在 `ckpt/MFSAN/multi_source/` 目录下：

```
ckpt/MFSAN/multi_source/
├── [CWRU_0_CWRU_1]To[CWRU_2]_MMDD-HHMMSS.log   # 训练日志
├── [CWRU_0_CWRU_1]To[CWRU_2]_MMDD-HHMMSS.pth   # 模型权重
├── training_curves.png                           # 训练曲线
└── tsne_visualization.png                        # t-SNE 可视化
```

### 日志解读

日志包含以下关键信息：
- `Train-Loss Source Classifier`: 源域分类损失
- `Train-Loss MMD`: MMD 分布对齐损失
- `Train-Loss L1`: 多源分类器一致性损失
- `Train-Acc Source Data`: 源域训练准确率
- `Val-acc`: 目标域验证准确率

## 可视化

### 训练曲线

```bash
python3 visualize_results.py [日志文件路径]
```

### t-SNE 特征可视化

```bash
python3 visualize_tsne_simple.py
```

这将生成两个 t-SNE 图：
1. 按故障类型着色的特征分布
2. 按域（源域/目标域）着色的特征分布

## 常见问题

### 1. Mat file appears to be empty

**问题**：读取 .mat 文件时出错

**解决**：确保数据文件是完整的 .mat 文件，排除 Zone.Identifier 等系统文件。代码已自动过滤。

### 2. ZeroDivisionError

**问题**：max_epoch=1 时除零错误

**解决**：使用 max_epoch >= 2

### 3. 数据集类别不匹配

**问题**：源域和目标域故障类别不一致

**解决**：确保每个 condition 文件夹包含相同名称的子文件夹

## 参考文献

- Zhu, Y., Zhuang, F., & Wang, D. (2019). Aligning domain-specific distribution and classifier for cross-domain classification from multiple sources. AAAI.
- 原始仓库：https://github.com/Liber1917/TL-Fault-Diagnosis-Library
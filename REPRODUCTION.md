# TL-Fault-Diagnosis-Library 复现指南

本指南帮助用户快速复现迁移学习故障诊断实验。

## 环境配置

### 1. Conda 环境创建

**Linux/macOS:**
```bash
# 创建 conda 环境
conda create -n tl-fault python=3.8 -y
conda activate tl-fault

# 安装 PyTorch (CPU 版本)
conda install pytorch==1.13.1 cpuonly -c pytorch -y

# 安装其他依赖
conda install numpy pandas scipy matplotlib scikit-learn tqdm -y
```

**Windows:**
```cmd
:: 创建 conda 环境
conda create -n tl-fault python=3.8 -y
conda activate tl-fault

:: 安装 PyTorch (CPU 版本)
conda install pytorch==1.13.1 cpuonly -c pytorch -y

:: 安装其他依赖
conda install numpy pandas scipy matplotlib scikit-learn tqdm -y
```

**或者使用 pip (通用):**
```bash
# Linux/macOS/Windows (在激活的 conda 环境中)
pip install torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy pandas scipy matplotlib scikit-learn tqdm
```

### 2. 克隆仓库

**Linux/macOS:**
```bash
git clone https://github.com/Liber1917/TL-Fault-Diagnosis-Library.git
cd TL-Fault-Diagnosis-Library
```

**Windows:**
```cmd
git clone https://github.com/Liber1917/TL-Fault-Diagnosis-Library.git
cd TL-Fault-Diagnosis-Library
```

---

## 数据集准备

### 数据集下载

从 GitHub Releases 下载示例数据集：
- 地址: https://github.com/Liber1917/TL-Fault-Diagnosis-Library/releases
- 下载: `Dataset-TL-FD-Library.zip`

### 数据集存放格式

#### 1. 域内迁移 (Within-Dataset Transfer)

同一数据集不同工况之间的迁移，例如 CWRU 的 condition_0 → condition_1。

**目录结构:**
```
datasets/
└── CWRU/
    ├── condition_0/    # 工况0: 0 HP, 1797 RPM
    │   ├── ball_07/    # 滚动体故障 7mil
    │   ├── ball_14/    # 滚动体故障 14mil
    │   ├── ball_21/    # 滚动体故障 21mil
    │   ├── inner_07/   # 内圈故障 7mil
    │   ├── inner_14/
    │   ├── inner_21/
    │   ├── outer_07/   # 外圈故障 7mil
    │   ├── outer_14/
    │   └── outer_21/
    ├── condition_1/    # 工况1: 1 HP, 1772 RPM
    ├── condition_2/    # 工况2: 2 HP, 1750 RPM
    └── condition_3/    # 工况3: 3 HP, 1730 RPM
```

**命名规则:** `数据集名_工况编号`，如 `CWRU_0`, `CWRU_1`

#### 2. 跨数据集迁移 (Cross-Dataset Transfer)

不同数据集之间的迁移，例如 CWRU → MFPT。

**目录结构:**
```
datasets/
├── CWRU/
│   ├── inner/      # 内圈故障
│   ├── normal/    # 正常状态
│   └── outer/     # 外圈故障
└── MFPT/
    ├── inner/
    ├── normal/
    └── outer/
```

**命名规则:** 直接使用数据集名称，如 `CWRU`, `MFPT`

#### 3. 选择性迁移 (Selective Transfer)

选择数据集中的特定类别进行迁移。

**命名规则:** `数据集名-类别编号`，如 `MFPT-01` 表示选择第0和第1个类别

### 数据文件格式

| 数据集 | 文件格式 | 说明 |
|--------|----------|------|
| CWRU | .mat | 振动信号数据 |
| MFPT | .mat | 振动信号数据 |
| PU | .mat | 振动信号数据 |
| XJTU | .csv | 振动信号数据 |
| IMS | .txt | 振动信号数据 |
| JNU | .csv | 振动信号数据 |

---

## 运行实验

### 1. 域内迁移 (One-to-One)

**Linux/macOS:**
```bash
# 激活环境
conda activate tl-fault

# 运行训练
python train.py \
    --model_name DAN \
    --source CWRU_0 \
    --target CWRU_1 \
    --train_mode single_source \
    --cuda_device '' \
    --max_epoch 30
```

**Windows:**
```cmd
conda activate tl-fault

python train.py --model_name DAN --source CWRU_0 --target CWRU_1 --train_mode single_source --cuda_device "" --max_epoch 30
```

### 2. 域内迁移 (Many-to-One)

```bash
python train.py \
    --model_name MFSAN \
    --source CWRU_0,CWRU_1 \
    --target CWRU_2 \
    --train_mode multi_source \
    --cuda_device '' \
    --max_epoch 30
```

### 3. 跨数据集迁移 (One-to-One)

```bash
python train.py \
    --model_name DAN \
    --source CWRU \
    --target MFPT \
    --train_mode single_source \
    --cuda_device '' \
    --max_epoch 30
```

### 4. 跨数据集迁移 (Many-to-One)

```bash
python train.py \
    --model_name MFSAN \
    --source CWRU,PU \
    --target MFPT \
    --train_mode multi_source \
    --cuda_device '' \
    --max_epoch 30
```

### 常用参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_name` | 模型名称 (DAN, CDAN, MFSAN, DANN, CORAL等) | CNN |
| `--source` | 源域，多个用逗号分隔 | CWRU_0 |
| `--target` | 目标域 | CWRU_1 |
| `--train_mode` | 训练模式: single_source / multi_source / source_combine | single_source |
| `--cuda_device` | GPU设备号，空字符串表示使用CPU | 0 |
| `--max_epoch` | 训练轮数 | 30 |
| `--batch_size` | 批次大小 | 64 |
| `--lr` | 学习率 | 0.01 |
| `--backbone` | 骨干网络: CNN 或 ResNet | CNN |
| `--signal_size` | 信号长度 (滑动窗口) | 1024 |

---

## 实验结果

训练完成后，结果保存在 `ckpt/<model_name>/<train_mode>/` 目录下：

```
ckpt/DAN/single_source/
├── [CWRU]To[MFPT]_MMDD-HHMMSS.log   # 训练日志
└── [CWRU]To[MFPT]_MMDD-HHMMSS.pth   # 模型权重
```

### 日志解读

- `Train-Loss Source Classifier`: 源域分类损失
- `Train-Loss Mk MMD` / `Train-Loss MMD`: MMD分布对齐损失 (DAN/CDAN)
- `Train-Loss Discriminator`: 判别器损失 (CDAN/DANN)
- `Train-Acc Source Data`: 源域训练准确率
- `Val-acc`: 目标域验证准确率

---

## 可视化

### 训练曲线

```bash
python visualize_results.py
```

### t-SNE 特征可视化

```bash
python visualize_tsne_simple.py
```

---

## 支持的模型

### 域适应 (Domain Adaptation)
- **DAN**: Deep Adaptation Networks
- **CDAN**: Conditional Adversarial Domain Adaptation
- **DANN**: Domain Adversarial Neural Network
- **CORAL**: Correlation Alignment
- **MFSAN**: Multi-Source Feature Alignment Network
- **MSSA**: Multi-Source Subdomain Adaptation

### 域泛化 (Domain Generalization)
- **IRM**: Invariant Risk Minimization
- **MixStyle**: Domain Generalization with MixStyle

### 部分/通用域适应
- **UDA**: Universal Domain Adaptation
- **IWAN**: Importance Weighted Adversarial Nets
- **AFN**: Adaptive Feature Norm

---

## 常见问题

### 1. Mat file appears to be empty

**问题**: 读取 .mat 文件时出错

**解决**: 
- 确保数据文件是完整的 .mat 文件
- Windows 下注意排除 Zone.Identifier 等系统文件
- 代码已自动过滤 `.mat` 文件

### 2. ZeroDivisionError

**问题**: max_epoch=1 时除零错误

**解决**: 使用 `max_epoch >= 2`

### 3. 数据集类别不匹配

**问题**: 源域和目标域故障类别不一致

**解决**: 
- 确保每个数据集文件夹包含相同名称的子文件夹
- 检查 `args.faults` 输出确认类别

### 4. CUDA 不可用

**问题**: 提示 CUDA 不可用

**解决**: 使用 `--cuda_device ''` 参数强制使用 CPU

### 5. Windows 路径问题

**问题**: Windows 下路径斜杠问题

**解决**: 代码已处理路径兼容性问题，使用 `os.path.join` 确保跨平台兼容

---

## 快速命令汇总

### Linux/macOS
```bash
# 完整实验流程
conda create -n tl-fault python=3.8 -y
conda activate tl-fault
conda install pytorch==1.13.1 cpuonly -c pytorch -y
conda install numpy pandas scipy matplotlib scikit-learn tqdm -y

git clone https://github.com/Liber1917/TL-Fault-Diagnosis-Library.git
cd TL-Fault-Diagnosis-Library

# 放置数据集到 datasets/ 目录

# 运行跨数据集迁移
python train.py --model_name DAN --source CWRU --target MFPT --train_mode single_source --cuda_device '' --max_epoch 30
```

### Windows
```cmd
:: 完整实验流程
conda create -n tl-fault python=3.8 -y
conda activate tl-fault
conda install pytorch==1.13.1 cpuonly -c pytorch -y
conda install numpy pandas scipy matplotlib scikit-learn tqdm -y

git clone https://github.com/Liber1917/TL-Fault-Diagnosis-Library.git
cd TL-Fault-Diagnosis-Library

:: 放置数据集到 datasets/ 目录

:: 运行跨数据集迁移
python train.py --model_name DAN --source CWRU --target MFPT --train_mode single_source --cuda_device "" --max_epoch 30
```

---

## 参考文献

- Long, M., et al. (2015). Learning transferable features with deep adaptation networks. ICML.
- Ganin, Y., et al. (2015). Unsupervised domain adaptation by backpropagation. ICML.
- Zhu, Y., et al. (2019). Aligning domain-specific distribution and classifier for cross-domain classification from multiple sources. AAAI.
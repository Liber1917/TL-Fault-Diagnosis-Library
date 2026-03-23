# QWEN.md - TL-Fault-Diagnosis-Library

## 项目概述

**TL-Fault-Diagnosis-Library** 是一个专注于跨域故障诊断的迁移学习库，支持**单源无监督域适应**、**多源无监督域适应**和**域泛化**。该库支持**闭集**、**开集**、**部分**和**通用**域适应场景。

### 主要技术栈

- **Python 3** (>=3.8)
- **PyTorch** (>=1.10)
- **NumPy**, **Pandas**, **SciPy**
- **matplotlib** (可视化)
- **scikit-learn** (t-SNE可视化)

---

## 项目结构

```
TL-Fault-Diagnosis-Library/
├── train.py              # 主训练脚本
├── opt.py                # 命令行参数配置
├── utils.py              # 工具函数 (GRL, MMD, 域判别器等)
├── train_utils.py        # 训练辅助函数
├── organize_cwru_data.py # CWRU数据集整理脚本
├── visualize_results.py  # 训练曲线可视化
├── visualize_tsne_simple.py # t-SNE特征可视化
├── requirements.txt      # 依赖列表
├── README.md             # 项目文档
├── REPRODUCTION.md       # 复现指南
│
├── models/               # 模型实现 (23个模型)
│   ├── __init__.py
│   ├── modules.py        # 共享模块
│   ├── CNN.py            # CNN backbone
│   │
│   ├── 域适应模型:
│   │   ├── ACDANN.py, ADACL.py, BSP.py, CDAN.py, CORAL.py
│   │   ├── DAN.py, DANN.py, MCD.py, MDD.py
│   │   ├── MFSAN.py, MSSA.py
│   │
│   ├── 域泛化模型:
│   │   ├── IRM.py, MixStyle.py, IBN.py, MLDG.py, DRO.py, VREx.py
│   │
│   └── 部分/通用域适应:
│       ├── IWAN.py, AFN.py, UDA.py
│
└── data_loader/          # 数据加载模块
    ├── load.py           # 主加载器
    ├── load_methods.py   # 数据集加载方法
    ├── data_utils.py     # 数据处理工具
    ├── conditional_load.py
    └── aug.py            # 数据增强
```

---

## 快速开始

### 1. 安装依赖

```bash
# 安装PyTorch (CPU版本)
pip3 install torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 安装其他依赖
pip3 install -r requirements.txt
pip3 install matplotlib scikit-learn
```

### 2. 数据集准备

创建 `datasets` 目录并组织数据：

```
datasets/
└── CWRU/
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

整理CWRU数据：
```bash
python3 organize_cwru_data.py
```

### 3. 训练模型

**单源域适应 (One-to-One):**
```bash
python train.py --model_name DAN --source CWRU_0 --target CWRU_1 --train_mode single_source --cuda_device 0
```

**多源域适应 (Many-to-One):**
```bash
python train.py --model_name MFSAN --source CWRU_0,CWRU_1 --target CWRU_2 --train_mode multi_source --cuda_device 0
```

**跨数据集迁移:**
```bash
python train.py --model_name MFSAN --source CWRU,PU --target MFPT --train_mode multi_source --cuda_device 0
```

---

## 核心命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_name` | 模型名称 (如 DAN, MFSAN, CDAN) | CNN |
| `--source` | 源域，多个用逗号分隔 | CWRU_0 |
| `--target` | 目标域 | CWRU_1 |
| `--train_mode` | 训练模式: single_source/source_combine/multi_source | single_source |
| `--cuda_device` | GPU设备号，空字符串表示CPU | 0 |
| `--max_epoch` | 训练轮数 | 30 |
| `--batch_size` | 批次大小 | 64 |
| `--lr` | 学习率 | 0.01 |
| `--backbone` | 骨干网络: CNN 或 ResNet | CNN |
| `--signal_size` | 信号长度 (滑动窗口) | 1024 |
| `--load_path` | 加载预训练模型路径 | - |

---

## 支持的模型

### 域适应 (Domain Adaptation)
- **ACDANN**, **ADACL**, **BSP**, **CDAN**, **CORAL**, **DAN**, **DANN**
- **MCD**, **MDD**, **MFSAN**, **MSSA**

### 域泛化 (Domain Generalization)
- **IRM**, **MixStyle**, **IBN**, **MLDG**, **GroupDRO**, **VREx**

### 部分/通用域适应
- **IWAN**, **AFN** (Partial DA)
- **UDA** (Universal DA)

---

## 可视化

### 训练曲线
```bash
python3 visualize_results.py
```

### t-SNE特征可视化
```bash
python3 visualize_tsne_simple.py
```

结果保存在 `ckpt/<model_name>/<train_mode>/` 目录下。

---

## 开发约定

1. **模型实现**: 每个模型在 `models/` 目录下创建独立的 `.py` 文件，需实现 `Trainer` 类
2. **数据加载**: 在 `data_loader/load_methods.py` 中添加新的数据集加载方法
3. **参数配置**: 所有超参数在 `opt.py` 中定义，使用 argparse
4. **训练日志**: 自动保存到 `ckpt/` 目录，包含 .log 和 .pth 文件

---

## 注意事项

- 域适应场景自动检测：代码会根据源域和目标域的类别关系自动判断为 closed-set、partial、open-set 或 universal 域适应
- 选择性迁移：使用 `dataset-numbers` 格式选择特定故障类别，如 `MFPT-01` 表示选择前两个类别
- 多源训练模式 (`source_combine`): 将多个源域合并为一个联合源域进行训练
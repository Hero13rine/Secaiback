# MIA: Membership Inference Attacks Against Object Detection Models

基于 Faster R-CNN 目标检测模型的成员推理攻击系统。

## 项目结构

```
MIA_detection/
├── pipeline.py             # 统一的MIA攻击流水线（推荐入口）
├── train_shadow.py         # 影子模型训练脚本
├── atk.py                  # 攻击模型训练脚本
├── mia.py                  # 成员推理攻击评估脚本
├── load_dataset.py         # 数据集加载工具
├── config.py               # 旧配置文件（已废弃，使用pipeline.py配置）
├── data/
│   └── dataset/
│       └── dior/
│           ├── train/      # 训练集（目标模型的成员样本）
│           ├── val/        # 验证集（影子模型的非成员样本）
│           └── test/       # 测试集（影子模型的训练集）
├── runs/
│   ├── shadow_train/       # 影子模型训练结果
│   └── attacker_train/     # 攻击模型训练结果
├── result/                 # 评估结果
├── requirements.txt
├── fasterrcnn_dior.pt      # 用户上传的目标模型
└── README.md
```

## 简化的MIA攻击流程

本项目实现了一个简化的成员推理攻击流程：

```
数据集分布:
├── TRAIN集 → 目标模型的训练数据（成员样本，用于最终评估）
├── VAL集   → 影子模型的非成员样本（用于攻击模型训练）
└── TEST集  → 影子模型的训练数据（成员样本，用于攻击模型训练）

攻击流程:
1. [已有] 目标模型在 TRAIN 集上训练
2. 影子模型在 TEST 集上训练（使用官方预训练权重）
3. 攻击模型训练：
   - 成员样本: TEST 集（影子模型见过）
   - 非成员样本: VAL 集（影子模型没见过）
4. 攻击评估：
   - 成员样本: 3000 张 TRAIN 集图片（目标模型见过）
   - 非成员样本: 3000 张 TEST 集图片（目标模型没见过）
```

## 安装依赖
secai-common环境可以直接用
```bash
pip install -r requirements.txt
```

## 快速开始

### 使用 Pipeline（推荐）

Pipeline 是统一的入口，可以在一个文件中配置所有参数：

```bash
# 运行完整流水线（配置检查 → 影子模型训练 → 攻击模型训练 → 评估）
python pipeline.py

# 只检查配置
python pipeline.py --steps 1

# 只训练影子模型
python pipeline.py --steps 2

# 只训练攻击模型
python pipeline.py --steps 3

# 只运行评估（需要已训练的模型）
python pipeline.py --steps 4
```

### 命令行参数覆盖

Pipeline 支持通过命令行参数覆盖默认配置：

```bash
# 指定 GPU
python pipeline.py --gpu 1

# 修改影子模型训练参数
python pipeline.py --shadow-epochs 50 --shadow-batch-size 8

# 修改攻击模型参数
python pipeline.py --attack-epochs 100 --attack-type shallow

# 修改评估样本数量
python pipeline.py --steps 4 --member-samples 5000 --nonmember-samples 5000

# 自定义数据路径
python pipeline.py --train-dir /path/to/train --val-dir /path/to/val --test-dir /path/to/test
```

### Pipeline 配置参数

在 `pipeline.py` 的 `PipelineConfig` 类中可以修改默认参数：

```python
@dataclass
class PipelineConfig:
    # 数据集路径
    TRAIN_DATA_DIR: str = './data/dataset/dior/train'
    VAL_DATA_DIR: str = './data/dataset/dior/val'
    TEST_DATA_DIR: str = './data/dataset/dior/test'

    # 模型路径
    TARGET_MODEL_DIR: str = 'runs/target_train/exp/best.pt'
    SHADOW_MODEL_DIR: str = 'runs/shadow_train/exp/best.pt'
    ATTACK_MODEL_DIR: str = 'runs/attacker_train/exp/best.pth'

    # 通用设置
    gpu_id: int = 0
    num_classes: int = 20
    img_size: int = 640

    # 影子模型训练
    SHADOW_EPOCHS: int = 30
    SHADOW_BATCH_SIZE: int = 16
    SHADOW_LR: float = 0.001
    SHADOW_USE_PRETRAINED: bool = True

    # 攻击模型训练
    ATTACK_EPOCHS: int = 80
    ATTACK_BATCH_SIZE: int = 32
    ATTACK_LR: float = 1e-5
    ATTACK_MODEL_TYPE: str = 'alex'  # 'alex' 或 'shallow'

    # Canvas/特征设置
    CANVAS_SIZE: int = 300
    MAX_LEN: int = 50
    LOG_SCORE: int = 2  # 0: raw, 1: ln, 2: log2
    CANVAS_TYPE: str = 'original'
    NORMALIZE_CANVAS: bool = True

    # MIA评估
    MIA_MEMBER_SAMPLES: int = 3000
    MIA_NONMEMBER_SAMPLES: int = 3000
```

## 单独运行各步骤

如需单独运行各步骤（不推荐），可以使用以下命令：

### 1. 训练影子模型

```bash
python train_shadow.py
```

### 2. 训练攻击模型

```bash
python atk.py --gpu_id 0 --batch_size 32
```

### 3. 评估攻击效果

```bash
python mia.py config
```

## 输出结果

### 训练输出
- 影子模型: `runs/shadow_train/exp/best.pt`
- 攻击模型: `runs/attacker_train/exp/best.pth`
- Canvas 可视化: `canvas_images/`

### 评估输出
- 攻击结果 CSV: `result/attack_results.csv`
- 评估指标 Pickle: `result/attack_evaluation_results.pkl`
- ROC 曲线: `result/attack_roc_curve.png`

### 评估指标说明
- **Accuracy**: 整体预测准确率
- **Precision**: 预测为成员的样本中真实成员的比例
- **Recall**: 正确识别的成员样本比例
- **F1**: Precision 和 Recall 的调和平均
- **AUC**: ROC 曲线下面积
- **TPR/FPR**: 真阳性率/假阳性率

## 数据集格式

数据集使用 YOLO 格式：

```
data/dataset/dior/
├── train/
│   ├── 00001.jpg
│   └── 00001.txt
├── val/
│   ├── 00002.jpg
│   └── 00002.txt
└── test/
    ├── 00003.jpg
    └── 00003.txt
```

标签文件格式（每行一个目标）：
```
<class_id> <cx> <cy> <width> <height>
```
其中坐标和尺寸都是归一化到 [0, 1] 的值。

## 注意事项

1. **目标模型**: 需要预先训练好目标模型并放置在 `runs/target_train/exp/best.pt`
2. **GPU 内存**: 如果显存不足，可以减小 batch size
3. **数据路径**: 确保数据集路径配置正确
4. **训练时间**: 完整流水线需要较长时间，建议使用 GPU

## 技术原理

本项目基于论文实现成员推理攻击：

1. **Shadow Model Training**: 训练一个与目标模型结构相同的影子模型
2. **Feature Extraction**: 使用模型对图像进行推理，提取边界框和置信度
3. **Canvas Generation**: 将检测结果转换为 2D 画布表示（热力图形式）
4. **Attack Model**: 训练一个 CNN 分类器区分成员和非成员样本
5. **Evaluation**: 在目标模型上评估攻击效果

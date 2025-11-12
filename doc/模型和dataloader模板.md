# 模型与 DataLoader 封装模板

本文介绍如何将模型封装为可由 `Estimator` 接管的形式，并提供一套可复用的数据加载器模板。

## 1. 模型封装模板（传递给 Estimator）

### 1.1 架构概览
- 业务模型需实现一个**纯模型包装类**，负责组合底层网络、前向推理与必要的后处理，最终由 `Estimator` 进行进一步封装。
- 推荐让包装类继承 `nn.Module` 或项目现有的基类（例如 `BaseModel`），确保可直接传入优化器、损失函数并无缝集成到训练/推理流程。
- 该包装类的主要职责：
  1. **初始化核心网络**：在 `__init__` 中构建或接收骨干网络、头部、损失等组件，并完成权重初始化。
  2. **定义前向逻辑**：通过 `forward` 暴露模型的标准接口，可根据任务在内部组织输出格式。
  3. **提供推理接口**：封装 `predict`（或复用 `forward`），输出可直接被 `Estimator` 消费的张量或字典。
  4. **准备 Estimator 所需的元信息**：如类别数、输入尺寸、后处理参数等，可通过属性或方法暴露。
- `EstimatorFactory.create` 在实例化时会将该模型包装类与损失、优化器等组合成 `BaseEstimator` 的实现。因此模型模板的目标是让 `Estimator` 能够“一键”接管，而无需再额外处理张量形状或设备切换。

### 1.2 代码模板
```python
# my_model.py
import torch
from torch import nn
from typing import Any, Dict, Tuple

class MyModel(nn.Module):
    """示例：可直接交给 Estimator 的分类模型封装"""

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        *,
        input_shape: Tuple[int, ...],
        num_classes: int,
        class_names: Tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_names = class_names or tuple(str(i) for i in range(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits

    @torch.inference_mode()
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """推理入口，可在此处增加 softmax、后处理等逻辑"""
        logits = self.forward(batch)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def get_metadata(self) -> Dict[str, Any]:
        """向 Estimator 暴露模型需要的元信息"""
        return {
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }
```

### 1.3 使用指南
1. **拆分组件**：将骨干网络（如 ResNet、Vision Transformer）与任务头（如 `nn.Linear` 分类头）拆分为可注入的子模块，方便在配置中复用或替换。
2. **封装前向逻辑**：`forward` 中只做张量变换；若推理需要额外处理（如解码框、NMS），放在 `predict` 中，保持训练与推理解耦。
3. **暴露元信息**：将输入尺寸、类别数、标签映射等放入 `get_metadata` 中，使 `EstimatorFactory` 能够自动补全配置。
4. **与 Estimator 对接**：
   ```python
   from estimator.estimator_factory import EstimatorFactory
   from model.my_model import MyModel

   model = MyModel(backbone, classifier, input_shape=(3, 224, 224), num_classes=1000)
   loss_fn = nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

   estimator_config = {
       "framework": "pytorch",
       "task": "classification",
       "parameters": {
           "input_shape": model.input_shape,
           "nb_classes": model.num_classes,
           "clip_values": (0.0, 1.0),
       },
   }

   estimator = EstimatorFactory.create(model, loss_fn, optimizer, estimator_config)
   preds = estimator.predict(batch)
   ```
5. **扩展任务场景**：检测、分割任务可在 `predict` 中返回 `Dict[str, torch.Tensor]`，`Estimator` 只需配置相应的 `task`，即可自动匹配包装实现。

---

## 2. DataLoader 封装模板

### 2.1 分类任务模板
```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def build_classification_loader(
    dataset_root: str,
    *,
    batch_size: int = 64,
    train: bool = False,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.247, 0.243, 0.261),
    num_workers: int = 4,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = datasets.CIFAR10(
        root=dataset_root,
        train=train,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )
```
> 若更换数据集，只需调整 `datasets.CIFAR10` 与归一化参数，其他逻辑可保持不变。

### 2.2 目标检测模板
```python
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple
import torch

class MyDetectionDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label_path = self.samples[idx]
        image = load_and_preprocess_image(image_path)  # 返回 torch.Tensor, CHW
        target = parse_yolo_label_file(label_path)     # 返回 {"boxes": ..., "labels": ...}
        return image, target

def detection_collate(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets)

def build_detection_loader(
    dataset: Dataset,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 4,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate,
    )
```
> 若对接 YOLO 系列，请在 `collate_fn` 中判断样本结构并适配 `img, targets, paths, shapes` 等输出，同时确保每个目标标注包含 `batch_id` 信息。

---

## 3. 开发者 checklist
- [ ] 编写模型包装类，确保 `forward` 与 `predict` 一致并返回 `Estimator` 需要的格式。
- [ ] 提供 `get_metadata`（或同等方式）暴露输入维度与类别信息。
- [ ] 为训练/验证数据集编写独立的 `DataLoader` 构造器，并根据任务定制 `collate_fn`。
- [ ] 在配置文件中更新新的模型、数据集参数模板，保证 `EstimatorFactory` 能正确读取。
- [ ] 编写最小化示例脚本验证模型与加载器能否独立运行，再交由 `Estimator` 托管。

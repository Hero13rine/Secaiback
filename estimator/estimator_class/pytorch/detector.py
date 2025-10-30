# estimator/estimator_class/pytorch/detector.py

from __future__ import annotations
from typing import Any, Iterable, Tuple, Optional, Dict
import numpy as np
import torch
from art.estimators.object_detection import PyTorchObjectDetector

class PyTorchObjectDetectionWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        loss: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        input_shape: Tuple[int, int, int] = (3, -1, -1),
        clip_values: Optional[Tuple[float, float]] = None,
        preprocessing: Tuple[float, float] = (0.0, 1.0),
        channels_first: bool = True,
        device_type: str = "cpu",
        preprocessing_defences: Optional[Iterable] = None,
        postprocessing_defences: Optional[Iterable] = None,
        attack_losses: Tuple[str, ...] = (
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ),
        **kwargs,
    ) -> None:
        self._loss = loss

        art_ctor_kwargs: Dict[str, Any] = {
            "model": model,
            "input_shape": input_shape,
            "optimizer": optimizer,
            "clip_values": clip_values,
            "channels_first": channels_first,
            "preprocessing_defences": preprocessing_defences,
            "postprocessing_defences": postprocessing_defences,
            "preprocessing": preprocessing,
            "attack_losses": attack_losses,
            "device_type": device_type,
        }

        # 初始化 ART 检测估计器
        self._detector = PyTorchObjectDetector(**art_ctor_kwargs)

        self._score_threshold = kwargs.get("score_threshold", None)

    def get_core(self):
        return self._detector

    @torch.no_grad()
    def predict(
        self,
        batch: torch.Tensor,
        score_threshold: Optional[float] = None,
    ):
        """
        返回 List[Dict]: 每张图一个 dict，含 numpy 数组：
          - 'boxes': (N, 4) float32 [x1,y1,x2,y2]
          - 'scores': (N,)  float32
          - 'labels': (N,)  int64
        """
        th = score_threshold if score_threshold is not None else self._score_threshold
        results = []
        # 逐张调用 ART 的 predict（其期望 batch 维度）
        for i in range(batch.shape[0]):
            preds_list = self._detector.predict(batch[i : i + 1])
            # ART 返回的是一个 list（长度=批大小），这里取第 0 个
            pred = preds_list[0]
            boxes: np.ndarray = pred["boxes"]
            scores: np.ndarray = pred["scores"]
            labels: np.ndarray = pred["labels"]

            if th is not None:
                keep = scores >= th
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            # 保证类型与测试断言一致
            results.append(
                {
                    "boxes": boxes.astype(np.float32, copy=False),
                    "scores": scores.astype(np.float32, copy=False),
                    "labels": labels.astype(np.int64, copy=False),
                }
            )
        return results

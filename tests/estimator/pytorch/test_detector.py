"""Tests for the PyTorch object detection estimator wrapper built on ART."""
from __future__ import annotations

import importlib
import sys
import unittest
from unittest import mock

try:
    import numpy as np
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - handled by unittest skip
    np = None
    torch = None
    nn = None


class DummyDetectionLoss(nn.Module):
    """
    最小可用的检测损失：
    - 可被调用（forward）
    - 返回一个标量张量
    - 提供 reduction 属性（很多校验会用到）
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, *args, **kwargs):
        # 返回一个需要梯度的 0.0 标量，满足类型/形状检查
        return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


@unittest.skipIf(np is None or torch is None, "numpy and torch are required for detector tests")
class PyTorchObjectDetectionWrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Reload the module so patches do not leak between tests when ART is missing.
        if "estimator.estimator_class.pytorch.detector" in sys.modules:
            del sys.modules["estimator.estimator_class.pytorch.detector"]
        module = importlib.import_module("estimator.estimator_class.pytorch.detector")
        self.detector_module = module
        self.Wrapper = module.PyTorchObjectDetectionWrapper

    def test_initializes_art_detector_with_expected_arguments(self):
        """
        测试 PyTorchObjectDetectionWrapper 是否正确初始化 ART 的 PyTorchObjectDetector。
        
        该测试验证：
        1. Wrapper 能正确接收并处理传入的参数
        2. 在初始化 PyTorchObjectDetector 时，不会传递 loss 参数（因为 loss 由 Wrapper 内部处理）
        3. 其他参数能正确传递给 PyTorchObjectDetector
        """
        model = torch.nn.Sequential()
        loss = DummyDetectionLoss()   # 传入一个"正儿八经"的损失
        optimizer = None

        with mock.patch.object(
            self.detector_module,
            "PyTorchObjectDetector",
            autospec=True,
        ) as mock_art_detector:
            instance = mock_art_detector.return_value

            wrapper = self.Wrapper(
                model,
                loss,                       # -> Wrapper 吃掉，不下传
                optimizer,
                input_shape=(3, 32, 32),
                clip_values=(0.0, 1.0),
                preprocessing=(0.5, 0.5),
                channels_first=True,
                device_type="cpu",
            )

        # 断言：调用 PyTorchObjectDetector 时不包含 loss
        mock_art_detector.assert_called_once()
        _, kwargs = mock_art_detector.call_args
        self.assertNotIn("loss", kwargs)

        self.assertEqual(kwargs["model"], model)
        self.assertEqual(kwargs["optimizer"], optimizer)
        self.assertEqual(kwargs["input_shape"], (3, 32, 32))
        self.assertEqual(kwargs["clip_values"], (0.0, 1.0))
        self.assertTrue(kwargs["channels_first"])
        self.assertEqual(kwargs["preprocessing"], (0.5, 0.5))
        self.assertEqual(kwargs["device_type"], "cpu")

        self.assertIs(wrapper.get_core(), instance)

    def test_predict_formats_and_filters_predictions(self):
        """
        测试预测结果的格式化和过滤功能。
        
        该测试验证：
        1. Wrapper 能正确调用 ART 的 PyTorchObjectDetector 进行预测
        2. 预测结果能被正确格式化为标准的字典格式（包含 boxes, scores, labels）
        3. 根据设定的 score_threshold 能正确过滤预测结果
        4. 对于低于阈值的预测结果能被正确过滤掉
        """
        model = torch.nn.Sequential()
        loss = DummyDetectionLoss()   # 同上：提供可用损失
        optimizer = None

        with mock.patch.object(
            self.detector_module,
            "PyTorchObjectDetector",
            autospec=True,
        ) as mock_art_detector:
            art_instance = mock_art_detector.return_value
            art_instance.predict.side_effect = [
                [
                    {
                        "boxes": np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
                        "scores": np.array([0.8], dtype=np.float32),
                        "labels": np.array([1], dtype=np.int64),
                    }
                ],
                [
                    {
                        "boxes": np.array([[0.5, 0.5, 1.5, 1.5]], dtype=np.float32),
                        "scores": np.array([0.4], dtype=np.float32),
                        "labels": np.array([2], dtype=np.int64),
                    }
                ],
            ]

            wrapper = self.Wrapper(
                model,
                loss,                   # -> Wrapper 吃掉，不下传
                optimizer,
                input_shape=(3, 16, 16),
                score_threshold=0.5,
            )

        batch = torch.zeros((2, 3, 16, 16), dtype=torch.float32)
        results = wrapper.predict(batch)

        self.assertEqual(len(results), 2)
        self.assertEqual(art_instance.predict.call_count, 2)

        first_boxes = results[0]["boxes"]
        self.assertTrue(np.allclose(first_boxes, np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)))
        self.assertTrue(np.allclose(results[0]["scores"], np.array([0.8], dtype=np.float32)))
        self.assertTrue(np.allclose(results[0]["labels"], np.array([1], dtype=np.int64)))

        self.assertEqual(results[1]["boxes"].shape[0], 0)
        self.assertEqual(results[1]["scores"].shape[0], 0)
        self.assertEqual(results[1]["labels"].shape[0], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
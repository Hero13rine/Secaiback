"""Entry script showcasing adversarial robustness evaluation for detection models."""
from __future__ import annotations

import importlib
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from estimator import EstimatorFactory
from metric.object_detection.fairness import evaluate_fairness_detection
from method import load_config
from model import load_model
from tests.fasterrcnn.load_dataset import load_data



def main(
    model_config_path: str = "/wkm/secai/secai-common/config/user/model_pytorch_det_fasterrcnn_fairness",
    num_workers: int = 0,
) -> None:
    print("进度: 开始执行公平性评估测试...")

    # 1.加载配置文件
    print("进度: 加载配置文件...")
    user_config = load_config(model_config_path)
    model_instantiation_config = user_config["model"]["instantiation"]
    model_estimator_config = user_config["model"]["estimator"]
    evaluation_config = user_config["evaluation"]
    print ("进度", "配置文件已加载完毕")

    print("进度: 初始化模型...")
    model = load_model(model_instantiation_config["model_path"], model_instantiation_config["model_name"]
                       , model_instantiation_config["weight_path"], model_instantiation_config["parameters"])
    print("进度", "模型初始化完成")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)
    loss = None

    print("进度: 创建估计器...")
    estimator = EstimatorFactory.create(
        model=model,
        loss=loss,
        optimizer=optimizer,
        config=model_estimator_config,
    )

    # 5.加载数据
    print("进度: 加载测试数据...")
    test_loader = load_data()
    print("进度", "数据集已加载")

    # 6.根据传入的评测类型进行评测
    print("开始执行检测流程测试...")

    print("进度: 准备公平性评估配置...")
    fairness_payload = (user_config)
    print("进度", "公平性评估配置准备完毕")
    print("进度: 运行公平性评估...")
    fairness_evaluator = evaluate_fairness_detection(estimator, test_loader, fairness_payload)
    print("检测流程测试完成。")
    print("进度: 公平性评估测试完成。")
if __name__ == "__main__":
    main()
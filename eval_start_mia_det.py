"""入口脚本：运行检测模型的成员推理攻击评测。"""
import os
import sys

import numpy as np
import torch
from torch import optim

from estimator import EstimatorFactory
from method import load_config
from model import load_model
# 修改导入语句，直接从 load_dataset 导入
from tests.fasterrcnn.load_dataset import load_data
from utils.sender import ConsoleResultSender as ResultSender
from metric.object_detection.safety.mia import evaluation_mia_detection


def main():
    user_id = "local_user"  # 本地调试时使用固定的用户ID
    model_id = "local_model"  # 本地调试时使用固定的模型ID
    evaluation_type = "fairness"  # 本地调试时使用固定的评测维度
    evaluation_path = "secai-common/config/user/model_pytorch_det_fasterrcnn_mia"  # 本地调试时使用本地配置文件路径

    # 1.加载配置文件
    user_config = load_config(evaluation_path)
    model_instantiation_config = user_config["model"]["instantiation"]
    model_estimator_config = user_config["model"]["estimator"]
    evaluation_config = user_config["evaluation"]
    task = model_estimator_config.get("task", "classification")
    ResultSender.send_log("进度", "配置文件已加载完毕")

    # 2.初始化模型
    model = load_model(
        model_instantiation_config["model_path"],
        model_instantiation_config["model_name"],
        model_instantiation_config["weight_path"],
        model_instantiation_config["parameters"],
    )
    ResultSender.send_log("进度", "模型初始化完成")

    # 3.获取优化器和损失函数

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)
    loss = None

    # 4.生成估计器
    estimator = EstimatorFactory.create(
        model=model,
        loss=loss,
        optimizer=optimizer,
        config=model_estimator_config,
    )
    ResultSender.send_log("进度", "估计器已生成")

    # 5.加载数据
    train_loader, val_loader, test_loader = load_data()
    ResultSender.send_log("进度", "数据集已加载")

    # 6.根据传入的评测类型进行评测
    evaluation_mia_detection(estimator, train_loader, val_loader, test_loader, evaluation_config["safety"])

    ResultSender.send_log("进度", "评测结束")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ResultSender.send_log("错误", str(e))

"""
测试fasterrcnn的评估流程
"""

import os
import sys

import numpy as np
import torch
from torch import optim

from estimator import EstimatorFactory
from method.load_config import load_config
from data.dummy_detection_dataset import DummyDetectionDataset
from metric.object_detection.basic.detection import cal_object_detection
# from update_table import update

from estimator import EstimatorFactory
from method import load_config
from model import load_model
# 直接从 load_dataset 导入dataloader
from fasterrcnn_test.load_dataset import load_data



def main():
    # 0、定义关键参数
    evaluation_path = "config/user/model_pytorch_det_fasterrcnn.yaml"
    # 1.加载配置文件
    user_config = load_config(evaluation_path)
    model_instantiation_config = user_config["model"]["instantiation"]
    model_estimator_config = user_config["model"]["estimator"]
    evaluation_config = user_config["evaluation"]
    print ("进度", "配置文件已加载完毕")

    # 2.初始化模型
    model = load_model(model_instantiation_config["model_path"], model_instantiation_config["model_name"]
                       , model_instantiation_config["weight_path"], model_instantiation_config["parameters"])
    print("进度", "模型初始化完成")

    # 3.获取优化器和损失函数
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)
    # TODO 目前的loss是None，在基础评测维度下，不需要使用loss构建估计器，暂时不知道在目标检测模型中的loss如何构建，因为loss是封装在模型中的，所以暂时无法使用loss构建估计器
    loss = None

    # 4.生成估计器
    estimator = EstimatorFactory.create(
        model=model,
        loss=loss,
        optimizer=optimizer,
        config=model_estimator_config
    )
    print("进度", "估计器已生成")

    # 5.加载数据
    test_loader = load_data("fasterrcnn_test/test")
    print("进度", "数据集已加载")

    # 6.根据传入的评测类型进行评测
    print("开始执行检测流程测试...")
    cal_object_detection(estimator, test_loader, evaluation_config)
    print("检测流程测试完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e))
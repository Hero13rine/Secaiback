import os
import sys

import numpy as np
import torch
from torch import optim

from estimator import EstimatorFactory
from utils import load_config
from model import load_model
# 修改导入语句，直接从 load_dataset 导入
from utils.load_dataloader import load_dataloader
from utils.sender import ResultSender

def main():
    # 将目标路径添加到系统路径
    sys.path.append('/app/userData/modelData/')
    sys.path.append('/app/systemData/database_code/')
    # 0.0获取当前 Pod 名称
    pod_name = os.getenv('HOSTNAME')  # 获取 Pod 名称（例如: 1242343443-1880539772613976065-basic）

    # 0.1从 Pod 名称中提取信息
    user_id = "local_user"  # 本地调试时使用固定的用户ID
    model_id = "local_model"  # 本地调试时使用固定的模型ID
    evaluation_type = "basic"  # 本地调试时使用固定的评测维度
    evaluation_path = "../config/user/model_pytorch_det.yaml"  # 本地调试时使用本地配置文件路径

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
        model_instantiation_config["parameters"]
    )
    ResultSender.send_log("进度", "模型初始化完成")

    # 3.获取优化器和损失函数
    if task == "detection":
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)
        loss = None
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss = torch.nn.CrossEntropyLoss()

    # 4.生成估计器
    estimator = EstimatorFactory.create(
        model=model,
        loss=loss,
        optimizer=optimizer,
        config=model_estimator_config,
    )
    ResultSender.send_log("进度", "估计器已生成")

    # 5.加载数据
    load_dataset = load_dataloader("../load_dataset.py")
    test_loader = load_dataset()
    ResultSender.send_log("进度", "数据集已加载")

    # 6.根据传入的评测类型进行评测
    # 基础维度
    from metric.object_detection.basic.basic import cal_basic
    if evaluation_type == "basic":
        cal_basic(estimator, test_loader, evaluation_config["basic"])

    ResultSender.send_log("进度", "评测结束")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ResultSender.send_log("错误", str(e))
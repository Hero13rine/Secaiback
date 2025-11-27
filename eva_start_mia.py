"""入口脚本：运行检测模型的成员推理攻击评测。"""
import os
import sys

import numpy as np

from method import load_config
from model import load_model
# 修改导入语句，直接从 load_dataset 导入
from fasterrcnn_test.load_mia_dataset import load_data
from utils.sender import ConsoleResultSender as ResultSender
from metric.object_detection.safety.mia import evaluation_mia_detection


def main():
    user_id = "local_user"  # 本地调试时使用固定的用户ID
    model_id = "local_model"  # 本地调试时使用固定的模型ID
    evaluation_type = "fairness"  # 本地调试时使用固定的评测维度
    evaluation_path = "config/user/model_pytorch_det_mia.yaml"  # 本地调试时使用本地配置文件路径

    # 1.加载配置文件
    user_config = load_config(evaluation_path)
    model_instantiation_config = user_config["model"]["instantiation"]
    evaluation_config = user_config["evaluation"]
    ResultSender.send_log("进度", "配置文件已加载完毕")

    # 2.初始化模型
    model = load_model(
        model_instantiation_config["model_path"],
        model_instantiation_config["model_name"],
        model_instantiation_config["weight_path"],
        model_instantiation_config["parameters"],
    )
    ResultSender.send_log("进度", "模型初始化完成")

    # 3.加载数据
    train_loader, val_loader, test_loader = load_data()
    ResultSender.send_log("进度", "数据集已加载")

    # 4.根据传入的评测类型进行评测
    safety_cfg = evaluation_config["safety"]
    # 将模型定义信息传递给 MIA 评测，便于影子模型加载
    safety_cfg.setdefault("model_path", model_instantiation_config.get("model_path", ""))
    safety_cfg.setdefault("model_name", model_instantiation_config.get("model_name", ""))
    safety_cfg.setdefault("model_parameters", model_instantiation_config.get("parameters", {}))
    safety_cfg["attack_config"] = evaluation_config.get("attack_config")

    evaluation_mia_detection(
        train_loader,
        val_loader,
        test_loader,
        safety_cfg,
        target_model=model,
        model_instantiation=model_instantiation_config,
    )

    ResultSender.send_log("进度", "评测结束")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ResultSender.send_log("错误", str(e))

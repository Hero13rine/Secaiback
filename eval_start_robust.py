import os

import torch

from estimator import EstimatorFactory
from fasterrcnn_test.load_dataset import load_data
from metric.object_detection.robustness import evaluation_robustness
from method import load_config
from model import load_model
from utils.sender import ResultSender

# 入口无需防御式编程，直接按照生产环境流程执行
pod_name = os.getenv("HOSTNAME", "")
parts = pod_name.split("-")
user_id = parts[0]
model_id = parts[1]
evaluation_type = parts[2]

evaluation_path = "/app/userData/modelData/evaluationConfigs/" + "evaluationConfig" + ".yaml"

try:
    ResultSender.send_log("进度", "鲁棒性评测入口启动")
    ResultSender.send_log(
        "进度",
        f"Pod: {pod_name}, 用户: {user_id}, 模型: {model_id}, 评测类型: {evaluation_type}",
    )

    user_config = load_config(evaluation_path)
    model_instantiation_config = user_config["model"]["instantiation"]
    model_estimator_config = user_config["model"]["estimator"]
    evaluation_config = user_config["evaluation"]["robustness"]
    ResultSender.send_log("进度", "配置文件加载完成")

    model = load_model(
        model_instantiation_config["model_path"],
        model_instantiation_config["model_name"],
        model_instantiation_config["weight_path"],
        model_instantiation_config["parameters"],
    )
    ResultSender.send_log("进度", "模型初始化完成")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)

    estimator = EstimatorFactory.create(
        model=model,
        loss=None,
        optimizer=optimizer,
        config=model_estimator_config,
    )
    ResultSender.send_log("进度", "估计器已生成")

    test_loader = load_data("fasterrcnn_test/test")
    ResultSender.send_log("进度", "测试数据集加载完成")

    ResultSender.send_log("进度", "开始执行鲁棒性评估")
    evaluation_robustness(
        estimator=estimator,
        test_data=test_loader,
        config={"robustness": evaluation_config},
        batch_size=1,
    )

    ResultSender.send_log("进度", "鲁棒性评估结束")
    ResultSender.send_status("成功")
except Exception as exc:
    ResultSender.send_log("错误", f"鲁棒性评测失败: {exc}")
    ResultSender.send_status("失败")
    raise

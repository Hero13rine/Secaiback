import os
import sys

import torch
from torch import optim

from metric.basic.basic import cal_basic
from metric.interpretability.shap.GradientShap import GradientShap
from utils.SecAISender import ResultSender
from metric.robustness.evaluate_robustness import evaluation_robustness

# 将目标路径添加到系统路径
sys.path.append('/app/userData/modelData/')
sys.path.append('/app/systemData/database_code/')
from update_table import update

from estimator import EstimatorFactory
from method import evaluate, load_config
from model import load_model
# 修改导入语句，直接从 load_dataset 导入
from load_dataset import load_data
from attack import AttackFactory


def main():

    # 0.0获取当前 Pod 名称
    pod_name = os.getenv('HOSTNAME')  # 获取 Pod 名称（例如: 1242343443-1880539772613976065-basic）

    # 0.1从 Pod 名称中提取信息
    parts = pod_name.split('-')  # 根据 '-' 分割名称
    user_id = parts[0]  # 第一部分是用户ID
    model_id = parts[1]  # 第二部分是模型ID
    evaluation_type = parts[2]  # 第三部分是评测维度，如"basic"，"robustness"
    evaluation_path = "/app/userData/modelData/evaluationConfigs/" + "evaluationConfig" + ".yaml"

    # 1.加载配置文件
    user_config = load_config(evaluation_path)
    model_instantiation_config = user_config["model"]["instantiation"]
    model_estimator_config = user_config["model"]["estimator"]
    evaluation_config = user_config["evaluation"]
    ResultSender.send_log("进度", "配置文件已加载完毕")

    # 2.初始化模型
    model = load_model(model_instantiation_config["model_path"], model_instantiation_config["model_name"]
                       , model_instantiation_config["weight_path"], model_instantiation_config["parameters"])
    ResultSender.send_log("进度", "模型初始化完成")

    # 3.获取优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss = torch.nn.CrossEntropyLoss()

    # 4.生成估计器
    estimator = EstimatorFactory.create(
        model=model,
        loss=loss,
        optimizer=optimizer,
        config=model_estimator_config
    )
    ResultSender.send_log("进度", "估计器已生成")

    # 5.加载数据
    test_loader = load_data()
    ResultSender.send_log("进度", "数据集已加载")

    # 6.根据传入的评测类型进行评测
    if evaluation_type == "basic":
        cal_basic(estimator, test_loader, evaluation_config["basic"])
    elif evaluation_type == "robustness":
        evaluation_robustness(estimator, test_loader, evaluation_config["robustness"])
    elif evaluation_type == "shap":
        GradientShap(model, test_loader)

    ResultSender.send_log("进度", "评测结束")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ResultSender.send_log("错误", str(e))
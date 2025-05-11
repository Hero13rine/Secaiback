import os

import torch
from torch import optim
import sys
# 将目标路径添加到系统路径
sys.path.append('/app/userData/modelData/')

from estimator import EstimatorFactory
from method import evaluate, load_config
from model import load_model
# 修改导入语句，直接从 load_dataset 导入
from load_dataset import load_cifar
from attack import AttackFactory


def main():

    # 0.0获取当前 Pod 名称
    pod_name = os.getenv('HOSTNAME')  # 获取 Pod 名称（例如: 1242343443-1880539772613976065-adversarialattack）

    # 0.1从 Pod 名称中提取信息
    parts = pod_name.split('-')  # 根据 '-' 分割名称
    user_id = parts[0]  # 第一部分是用户ID
    model_id = parts[1]  # 第二部分是模型ID
    attack_type = parts[2]  # 第三部分是评测类型，如 "backdoorAttack" 或 "adversarialAttack"
    evaluation_path = "/app/userData/modelData/evaluationConfigs/" + attack_type + ".yaml"

    # 1.加载配置文件
    user_config = load_config(evaluation_path)
    model_instantiation_config = user_config["model"]["instantiation"]
    model_estimator_config = user_config["model"]["estimator"]
    attack_config = user_config["attack"]

    # 2.初始化模型
    model = load_model(model_instantiation_config["model_path"], model_instantiation_config["model_name"]
                       , model_instantiation_config["weight_path"], model_instantiation_config["parameters"])

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


    # 5.生成攻击对象
    attack = AttackFactory.create(
        estimator=estimator.get_core(),
        config=attack_config
    )

    # 6.加载数据
    test_loader = load_cifar()

    # 7.进行评估
    evaluate(test_loader, estimator, attack)


if __name__ == "__main__":
    main()
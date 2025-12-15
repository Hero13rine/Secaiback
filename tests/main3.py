import torch
import yaml
import numpy as np
from torch import optim

from estimator import EstimatorFactory
from method import evaluate, load_config, load_system_config
from model import load_model
from data.load_dataset import load_mnist, load_cifar
from attack import AttackFactory
from utils.visualize import plot_samples


def main():

    # 1.加载配置文件
    system_config = load_system_config("config/system/system_config.yaml")
    model_config, attack_config = load_config(system_config["model"]["config_path"], system_config["attack"]["config_path"])

    # 2.初始化模型
    model = load_model(model_config["model_path"], model_config["model_name"], model_config["weight_path"])

    # 3.获取优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss = torch.nn.CrossEntropyLoss()

    # 4.生成估计器
    estimator = EstimatorFactory.create(
        framework=model_config["framework"],
        task=model_config["task"],
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        loss=loss,
        optimizer=optimizer,
        input_shape=tuple(model_config["input_shape"]),
        nb_classes=model_config["num_classes"],
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )

    # 5.生成攻击对象
    attack = AttackFactory.create(
        name=attack_config["attack"]["name"],
        estimator=estimator.get_core(),
        **attack_config["attack"]["params"]
    )

    # 6.加载数据
    test_loader = load_cifar()

    # 7.进行评估
    evaluate(test_loader, estimator, attack)


if __name__ == "__main__":
    main()
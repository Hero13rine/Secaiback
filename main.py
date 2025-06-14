import torch
from art.utils import load_cifar10
from torch import optim

from estimator import EstimatorFactory
from method import evaluate, load_config, evaluate_robustness
from model import load_model
from data.load_dataset import load_cifar
from attack import AttackFactory


def main():

    # 1.加载配置文件
    user_config = load_config("config/user/model_pytorch_cls_cw.yaml")
    # user_config = load_config("config/user/java_generate_test.yaml")
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
    evaluate_robustness(test_loader, estimator, attack)


if __name__ == "__main__":
    main()
import torch
from art.utils import load_cifar10
from torch import optim

from estimator import EstimatorFactory
from method import evaluate, load_config, evaluate_robustness_adv, evaluate_robustness_corruptions, evaluate_clean, \
    evaluation_robustness, evaluate_robustness_adv_all
from model import load_model
from data.load_dataset import load_cifar
from attack import AttackFactory


def main():
    # 1.加载配置文件
    user_config = load_config("config/user/model_pytorch_cls.yaml")
    # user_config = load_config("config/user/java_generate_test.yaml")
    model_instantiation_config = user_config["model"]["instantiation"]
    model_estimator_config = user_config["model"]["estimator"]
    evaluation_config = user_config["evaluation"]

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

    # 6.加载数据
    test_loader = load_cifar()

    if len(evaluation_config["robustness"]["adversarial"]) > 0 or len(
            evaluation_config["robustness"]["corruption"]) > 0:
        metrics = evaluation_config["robustness"]
        evaluation_robustness(test_loader, estimator, metrics)

    # 7.进行评估
    # evaluate_robustness_adv(test_loader, estimator, attack)
    # evaluate_robustness_adv_all(test_loader,estimator)
    # evaluate_robustness_corruptions(test_loader,estimator)
    # evaluate_clean(test_loader, estimator)


if __name__ == "__main__":
    main()

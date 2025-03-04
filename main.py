import torch
import yaml
import numpy as np
from art.estimators.classification import PyTorchClassifier

from model.net.cifar10 import Tudui
from data.load_dataset import load_mnist, load_cifar
from attack import AttackFactory
from utils.visualize import plot_samples


def main():
    # 加载配置
    with open("config/model_config.yaml") as f:
        model_config = yaml.safe_load(f)
    with open("config/attack_config.yaml") as f:
        attack_config = yaml.safe_load(f)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tudui().to(device)
    model.load_state_dict(torch.load(model_config["model"]["weight_path"]))
    model.eval()

    # 包装为ART分类器
    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters()),
        input_shape=tuple(model_config["model"]["input_shape"]),
        nb_classes=model_config["model"]["num_classes"],
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )

    # 生成对抗样本
    attack = AttackFactory.create(
        name=attack_config["attack"]["name"],
        estimator=classifier,
        **attack_config["attack"]["params"]
    )

    # 加载数据
    test_loader = load_cifar()
    total_correct_clean = 0
    total_correct_adv = 0
    total_samples = 0

    for x_batch, y_batch in test_loader:
        x_batch_np = x_batch.numpy()
        y_batch_np = y_batch.numpy()

        # 原始预测
        pred_clean = classifier.predict(x_batch_np)
        total_correct_clean += np.sum(np.argmax(pred_clean, axis=1) == y_batch_np)

        # 生成对抗样本
        x_adv_np = attack.generate(x_batch_np)

        # 对抗样本预测
        pred_adv = classifier.predict(x_adv_np)
        total_correct_adv += np.sum(np.argmax(pred_adv, axis=1) == y_batch_np)

        total_samples += len(y_batch)

        # 可视化对比
        # plot_samples(x_batch_np, x_adv_np, y_batch_np)

    # 计算整体准确率
    acc_clean = total_correct_clean / total_samples
    acc_adv = total_correct_adv / total_samples
    print(f"Clean accuracy (full test set): {acc_clean:.2%}")
    print(f"Adversarial accuracy (full test set): {acc_adv:.2%}")


if __name__ == "__main__":
    main()
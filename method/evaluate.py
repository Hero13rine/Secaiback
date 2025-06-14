# 加载数据
import numpy as np
import torch

from data.load_dataset import load_cifar


def evaluate(test_loader, estimator, attack):
    total_correct_clean = 0
    total_correct_adv = 0
    total_samples = 0
    for x_batch, y_batch in test_loader:
        x_batch_np = x_batch.numpy()
        y_batch_np = y_batch.numpy()

        # 原始预测
        pred_clean = estimator.predict(x_batch_np)
        total_correct_clean += np.sum(np.argmax(pred_clean, axis=1) == y_batch_np)

        # 生成对抗样本
        x_adv_np = attack.generate(x_batch_np)

        # 对抗样本预测
        pred_adv = estimator.predict(x_adv_np)
        total_correct_adv += np.sum(np.argmax(pred_adv, axis=1) == y_batch_np)

        total_samples += len(y_batch)

        # 可视化对比
        # plot_samples(x_batch_np, x_adv_np, y_batch_np)

    # 计算整体准确率
    acc_clean = total_correct_clean / total_samples
    acc_adv = total_correct_adv / total_samples
    print(f"Clean accuracy (full test set): {acc_clean:.2%}")
    print(f"Adversarial accuracy (full test set): {acc_adv:.2%}")
    return acc_clean, acc_adv


# 定义softmax函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def evaluate_robustness(test_loader, estimator, attack):
    total_correct_clean = 0
    total_correct_adv = 0
    total_samples = 0
    successful_attack_confidences = []  # 用于存储攻击成功样本的真实类别置信度
    acac_confidences = []  # 用于存储攻击成功样本的错误类别置信度

    for x_batch, y_batch in test_loader:
        x_batch_np = x_batch.numpy().astype(np.float32)
        y_batch_np = y_batch.numpy()

        # 原始预测
        pred_clean = estimator.predict(x_batch_np)

        # 对原始预测进行softmax归一化
        pred_clean_probs = softmax(pred_clean)
        total_correct_clean += np.sum(np.argmax(pred_clean_probs, axis=1) == y_batch_np)

        # 生成对抗样本
        x_adv_np = attack.generate(x_batch_np)
        # print(x_adv_np.dtype)

        # 对抗样本预测
        pred_adv = estimator.predict(x_adv_np)
        # 对对抗样本预测进行softmax归一化
        pred_adv_probs = softmax(pred_adv)
        total_correct_adv += np.sum(np.argmax(pred_adv_probs, axis=1) == y_batch_np)

        # 找出攻击成功的样本
        attack_success = np.argmax(pred_adv_probs, axis=1) != y_batch_np
        for i in range(len(y_batch_np)):
            if attack_success[i]:
                # 提取攻击成功样本的真实类别置信度
                true_class_confidence = pred_adv_probs[i][y_batch_np[i]]
                successful_attack_confidences.append(true_class_confidence)

                # 提取攻击成功样本的错误类别置信度
                misclassified_class = np.argmax(pred_adv_probs[i])
                misclassified_confidence = pred_adv_probs[i][misclassified_class]
                acac_confidences.append(misclassified_confidence)

        total_samples += len(y_batch)

    # 计算整体准确率
    acc_clean = total_correct_clean / total_samples
    acc_adv = total_correct_adv / total_samples
    print(f"Clean accuracy (full test set): {acc_clean:.2%}")
    print(f"Adversarial accuracy (full test set): {acc_adv:.2%}")

    # 计算ACTC
    if len(successful_attack_confidences) > 0:
        actc = np.mean(successful_attack_confidences)
        print(f"ACTC (Average Confidence of True Class): {actc:.4f}")
    else:
        print("No successful attacks found. ACTC cannot be calculated.")
        actc = None

    # 计算ACAC
    if len(acac_confidences) > 0:
        acac = np.mean(acac_confidences)
        print(f"ACAC (Average Confidence of Adversarial Class): {acac:.4f}")
    else:
        print("No successful attacks found. ACAC cannot be calculated.")
        acac = None

    return acc_clean, acc_adv, actc, acac

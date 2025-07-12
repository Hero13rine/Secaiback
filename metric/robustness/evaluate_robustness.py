# 加载数据
import numpy as np
import torch
from torchvision import transforms

from attack import AttackFactory
from utils.SecAISender import ResultSender

from method.corruptions import (
    gaussian_noise, shot_noise, impulse_noise, speckle_noise,
    gaussian_blur, glass_blur, defocus_blur, motion_blur, zoom_blur,
    fog, frost, snow, spatter, contrast, brightness, saturate,
    jpeg_compression, pixelate, elastic_transform
)


# 定义softmax函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def evaluation_robustness(test_loader, estimator, metrics):
    ResultSender.send_log("进度", "鲁棒性评测开始")
    print("鲁棒性评测开始")
    metrics_adv = metrics["adversarial"]
    if len(metrics_adv) > 0:
        evaluate_robustness_adv_all(test_loader, estimator, metrics_adv)
    metrics_cor = metrics["corruption"]
    if len(metrics_cor) > 0:
        evaluate_robustness_corruptions(test_loader, estimator, metrics_cor)


def evaluate_robustness_adv(test_loader, estimator, attack):
    total_uncorrect_adv = 0
    total_samples = 0
    successful_attack_confidences = []  # 用于存储攻击成功样本的真实类别置信度
    acac_confidences = []  # 用于存储攻击成功样本的错误类别置信度

    for x_batch, y_batch in test_loader:
        x_batch_np = x_batch.numpy().astype(np.float32)
        y_batch_np = y_batch.numpy()

        # 生成对抗样本
        x_adv_np = attack.generate(x_batch_np)

        # 对抗样本预测
        pred_adv = estimator.predict(x_adv_np)
        # 对对抗样本预测进行softmax归一化
        pred_adv_probs = softmax(pred_adv)
        total_uncorrect_adv += np.sum(np.argmax(pred_adv_probs, axis=1) != y_batch_np)

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
    adverr = total_uncorrect_adv / total_samples
    advacc = 1 - adverr
    # ResultSender.send_result("adverr",adverr)
    # ResultSender.send_result("advacc", advacc)
    print(f"Adversarial  dataset accuracy (full test set): {advacc:.2%}")
    print(f"Adversarial  dataset error (full test set): {adverr:.2%}")

    # 计算ACTC
    if len(successful_attack_confidences) > 0:
        actc = np.mean(successful_attack_confidences)
        # ResultSender.send_result("actc", actc)
        print(f"actc (Average Confidence of True Class): {actc:.4f}")
    else:
        ResultSender.send_log("异常", "No successful attacks found. actc cannot be calculated.")
        print("No successful attacks found. actc cannot be calculated.")
        actc = None

    # 计算ACAC
    if len(acac_confidences) > 0:
        acac = np.mean(acac_confidences)
        # ResultSender.send_result("acac", acac)
        print(f"acac (Average Confidence of Adversarial Class): {acac:.4f}")
    else:
        # ResultSender.send_log("异常", "No successful attacks found. acac cannot be calculated.")
        print("No successful attacks found. acac cannot be calculated.")
        acac = None

    return adverr, advacc, actc, acac


def parse_attack_method(attack_str):
    """将攻击方法字符串解析为包含方法和参数的字典"""
    return {
        "method": attack_str,
        "parameters": {
            "eps": 0.4
        }
    }


def evaluate_robustness_adv_all(test_loader, estimator, metrics):
    ResultSender.send_log("进度", "对抗攻击评测开始")
    # attack_method = ["fgsm","pgd","cw0"]
    attack_method = ["fgsm"]

    # 存储所有攻击方法的结果
    all_results = {
        'adverr': [],
        'advacc': [],
        'actc': [],
        'acac': []
    }

    # 对每种攻击方法进行评估
    for attack_name in attack_method:
        attack_config = parse_attack_method(attack_name)
        attack = AttackFactory.create(
            estimator=estimator.get_core(),
            config=attack_config
        )
        adverr, advacc, actc, acac = evaluate_robustness_adv(test_loader, estimator, attack)

        # 收集每种攻击方法的结果
        all_results['adverr'].append(adverr)
        all_results['advacc'].append(advacc)
        all_results['actc'].append(actc)
        all_results['acac'].append(acac)

    # 计算所有指标的平均值
    if "adverr" in metrics:
        print(f"adverr: {sum(all_results['adverr'])/len(all_results['adverr'])}")
        ResultSender.send_result("adverr", sum(all_results['adverr'])/len(all_results['adverr']))
    if "advacc" in metrics:
        print(f"advacc: {sum(all_results['advacc']) / len(all_results['advacc'])}")
        ResultSender.send_result("advacc", sum(all_results['advacc']) / len(all_results['advacc']))
    if "actc" in metrics:
        print(f"actc: {sum(all_results['actc']) / len(all_results['actc'])}")
        ResultSender.send_result("actc", sum(all_results['actc']) / len(all_results['actc']))
    if "acac" in metrics:
        print(f"acac: {sum(all_results['acac']) / len(all_results['acac'])}")
        ResultSender.send_result("acac", sum(all_results['acac']) / len(all_results['acac']))


def evaluate_clean(test_loader, estimator):
    total_correct_clean = 0
    total_samples = 0

    for x_batch, y_batch in test_loader:
        x_batch_np = x_batch.numpy().astype(np.float32)
        y_batch_np = y_batch.numpy()

        # 原始预测
        pred_clean = estimator.predict(x_batch_np)

        # 对原始预测进行softmax归一化
        pred_clean_probs = softmax(pred_clean)
        total_correct_clean += np.sum(np.argmax(pred_clean_probs, axis=1) != y_batch_np)

        total_samples += len(y_batch)

    # 计算整体准确率
    err_clean = 100 * total_correct_clean / total_samples
    print(f"asr_clean (full test set): {err_clean:.2f}%")
    return err_clean


def evaluate_robustness_corruptions(test_loader, estimator, metrics):
    ResultSender.send_log("进度", "扰动攻击评测开始")
    # 定义所有扰动方法
    corruption_functions = [
        gaussian_noise, shot_noise, impulse_noise, speckle_noise,
        gaussian_blur, glass_blur, defocus_blur, motion_blur, zoom_blur,
        fog, frost, snow, spatter, contrast, brightness, saturate,
        jpeg_compression, pixelate, elastic_transform
    ]
    # 定义要测试的 severity 级别
    severity_levels = [1, 2, 3, 4, 5]

    asr_total = 0

    # 遍历所有扰动方法
    for corruption_function in corruption_functions:
        for severity in severity_levels:
            total = 0
            un_correct = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    for i in range(images.size(0)):
                        image = images[i].permute(1, 2, 0).numpy() * 255
                        image = image.astype(np.uint8)
                        # 应用扰动
                        corrupted_image = corruption_function(image, severity=severity)
                        if isinstance(corrupted_image, np.ndarray):
                            corrupted_image = torch.from_numpy(corrupted_image / 255.0).permute(2, 0, 1).float()
                        else:
                            corrupted_image = transforms.ToTensor()(corrupted_image)
                        corrupted_image = corrupted_image.unsqueeze(0)
                        # 前向传播
                        outputs = estimator.predict(corrupted_image)
                        # 使用 NumPy 找到预测的类别
                        predicted = np.argmax(outputs, axis=1)
                        total += 1
                        un_correct += (predicted[0] != labels[i].item())
            err_corruption = 100 * un_correct / total
            ResultSender.send_log("进度",
                                  f"UnCorrectNum of the network on the test images with {corruption_function.__name__}_{severity}: {un_correct}")
            ResultSender.send_log("进度",
                                  f"ASR of the network on the test images with {corruption_function.__name__}_{severity}: {err_corruption:.2f}%")
            print(
                f"UnCorrectNum of the network on the test images with {corruption_function.__name__}_{severity}: {un_correct}")
            print(
                f"ASR of the network on the test images with {corruption_function.__name__}_{severity}: {err_corruption:.2f}%")
            asr_total += err_corruption
    mCE = asr_total / (len(corruption_functions) * len(severity_levels))
    print(f"mCE of the network on the test images: {mCE:.2f}%")
    if "mce" in metrics:
        ResultSender.send_result("mce", mCE)
    if "rmce" in metrics:
        err_clean = evaluate_clean(test_loader, estimator)
        RmCE = mCE - err_clean
        print(f"RmCE of the network on the test images: {RmCE:.2f}%")
        ResultSender.send_result("rmce", RmCE)

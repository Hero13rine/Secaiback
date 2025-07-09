# 加载数据
import numpy as np
import torch
from torchvision import transforms
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


def evaluation_robustness(test_loader, estimator, attack):
    ResultSender.send_log("进度","鲁棒性评测开始")
    evaluate_robustness_adv(test_loader, estimator, attack)
    evaluate_robustness_corruptions(test_loader, estimator)


def evaluate_robustness_adv(test_loader, estimator, attack):
    ResultSender.send_log("进度","对抗攻击评测开始")
    total_correct_clean = 0
    total_uncorrect_adv = 0
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
    acc_clean = total_correct_clean / total_samples
    ASR = total_uncorrect_adv / total_samples
    ResultSender.send_result("acc_clean",acc_clean)
    ResultSender.send_result("ASR", ASR)
    print(f"Clean accuracy (full test set): {acc_clean:.2%}")
    print(f"Attack accuracy (full test set): {ASR:.2%}")

    # 计算ACTC
    if len(successful_attack_confidences) > 0:
        actc = np.mean(successful_attack_confidences)
        ResultSender.send_result("ACTC", actc)
        print(f"ACTC (Average Confidence of True Class): {actc:.4f}")
    else:
        ResultSender.send_log("异常", "No successful attacks found. ACTC cannot be calculated.")
        print("No successful attacks found. ACTC cannot be calculated.")
        actc = None

    # 计算ACAC
    if len(acac_confidences) > 0:
        acac = np.mean(acac_confidences)
        ResultSender.send_result("ACAC", acac)
        print(f"ACAC (Average Confidence of Adversarial Class): {acac:.4f}")
    else:
        ResultSender.send_log("异常", "No successful attacks found. ACAC cannot be calculated.")
        print("No successful attacks found. ACAC cannot be calculated.")
        acac = None

    return acc_clean, ASR, actc, acac


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
    ResultSender.send_result("err_clean", err_clean)
    print(f"asr_clean (full test set): {err_clean:.2f}%")

    return err_clean


def evaluate_robustness_corruptions(test_loader, estimator):
    ResultSender.send_log("进度","扰动攻击评测开始")
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
            ResultSender.send_log("进度",  f"UnCorrectNum of the network on the test images with {corruption_function.__name__}_{severity}: {un_correct}")
            ResultSender.send_log("进度",
                                   f"ASR of the network on the test images with {corruption_function.__name__}_{severity}: {err_corruption:.2f}%")
            print(
                f"UnCorrectNum of the network on the test images with {corruption_function.__name__}_{severity}: {un_correct}")
            print(
                f"ASR of the network on the test images with {corruption_function.__name__}_{severity}: {err_corruption:.2f}%")
            asr_total += err_corruption
    mCE = asr_total / (len(corruption_functions) * len(severity_levels))
    ResultSender.send_result("mCE", mCE)
    print(f"mCE of the network on the test images: {mCE:.2f}%")
    err_clean = evaluate_clean(test_loader, estimator)
    RmCE = mCE - err_clean
    ResultSender.send_result("RmCE", RmCE)
    print(f"RmCE of the network on the test images: {RmCE:.2f}%")
    return mCE, RmCE

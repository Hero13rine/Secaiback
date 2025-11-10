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
    try:
        metrics_adv = metrics["adversarial"]
        if len(metrics_adv) > 0:
            evaluate_robustness_adv_all(test_loader, estimator, metrics_adv)
        metrics_cor = metrics["corruption"]
        if len(metrics_cor) > 0:
            evaluate_robustness_corruptions(test_loader, estimator, metrics_cor)
        ResultSender.send_status("成功")
        ResultSender.send_log("进度", "评测结果已写回数据库")
    except Exception as e:
        ResultSender.send_status("失败")
        ResultSender.send_log("错误", str(e))

def process_predictions(images_np, estimator):
    """统一处理4维和5维数据的预测逻辑"""
    if len(images_np.shape) == 5:  # (bs, ncrops, c, h, w) 10折裁剪数据
        bs, ncrops, c, h, w = images_np.shape
        images_flat = images_np.reshape(-1, c, h, w)  # 展平裁剪维度
        outputs = estimator.predict(images_flat)
        outputs_avg = outputs.reshape(bs, ncrops, -1).mean(axis=1)  # 平均融合
        return outputs_avg
    elif len(images_np.shape) == 4:  # (bs, c, h, w) 单图数据
        return estimator.predict(images_np)
    else:
        raise ValueError(f"不支持的数据维度: {images_np.shape}，仅支持4维或5维")


def evaluate_robustness_adv(test_loader, estimator, attack):
    total_uncorrect_adv = 0
    total_samples = 0
    successful_attack_confidences = []  # 用于存储攻击成功样本的真实类别置信度
    acac_confidences = []  # 用于存储攻击成功样本的错误类别置信度

    for x_batch, y_batch in test_loader:
        x_batch_np = x_batch.numpy().astype(np.float32)
        y_batch_np = y_batch.numpy()
        bs = y_batch_np.shape[0]

        # 生成对抗样本
        if len(x_batch_np.shape) == 5:
            # 5维数据先展平裁剪维度生成对抗样本
            bs_adv, ncrops_adv, c_adv, h_adv, w_adv = x_batch_np.shape
            x_flat = x_batch_np.reshape(-1, c_adv, h_adv, w_adv)
            x_adv_flat = attack.generate(x_flat)
            x_adv_np = x_adv_flat.reshape(bs_adv, ncrops_adv, c_adv, h_adv, w_adv)
        else:
            x_adv_np = attack.generate(x_batch_np)

        # 对抗样本预测
        pred_adv = process_predictions(x_adv_np, estimator)
        # 对对抗样本预测进行softmax归一化
        pred_adv_probs = softmax(pred_adv)
        total_uncorrect_adv += np.sum(np.argmax(pred_adv_probs, axis=1) != y_batch_np)

        # 找出攻击成功的样本
        attack_success = np.argmax(pred_adv_probs, axis=1) != y_batch_np
        for i in range(bs):
            if attack_success[i]:
                # 提取攻击成功样本的真实类别置信度
                true_class_confidence = pred_adv_probs[i][y_batch_np[i]]
                successful_attack_confidences.append(true_class_confidence)

                # 提取攻击成功样本的错误类别置信度
                misclassified_class = np.argmax(pred_adv_probs[i])
                misclassified_confidence = pred_adv_probs[i][misclassified_class]
                acac_confidences.append(misclassified_confidence)

        total_samples += bs

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


def parse_attack_method(attack_str, eps):
    """将攻击方法字符串解析为包含方法和参数的字典"""
    return {
        "method": attack_str,
        "parameters": {
            "eps": eps
        }
    }


def evaluate_robustness_adv_all(test_loader, estimator, metrics):
    ResultSender.send_log("进度", "对抗攻击评测开始")
    attack_method = ["fgsm"]

    # eps参数列表，从0到1，步长0.1
    eps_list = [round(eps, 1) for eps in np.arange(0.0, 1.1, 0.1)]

    # 存储所有eps的结果
    eps_results = {}  # 键是eps，值是包含各指标的字典

    # 对每种攻击方法和eps值进行评估
    for attack_name in attack_method:
        for eps in eps_list:
            print(f"\nEvaluating {attack_name} with eps={eps}")
            attack_config = parse_attack_method(attack_name, eps)
            attack = AttackFactory.create(
                estimator=estimator.get_core(),
                config=attack_config
            )
            adverr, advacc, actc, acac = evaluate_robustness_adv(test_loader, estimator, attack)

            # 存储当前eps的结果
            eps_results[eps] = {
                'adverr': adverr,
                'advacc': advacc,
                'actc': actc,
                'acac': acac
            }

            # 遍历每个指标，每次只发送一个键值对，键名格式为 metric_eps（如 acac_0_1）
            for metric in metrics:
                value = eps_results[eps][metric]
                # 将eps中的小数点替换为下划线（如0.1 → 0_1）
                eps_str = str(eps).replace('.', '_')
                key = f"{metric}_{eps_str}"
                # 单个键值对传递：第一个参数是key，第二个是value
                if value is not None:
                    ResultSender.send_result(key, f"{value:.4f}")
                else:
                    ResultSender.send_result(key, "None")

    # 计算所有eps的平均值，同样每次发送一个键值对
    avg_results = {}
    for metric in metrics:
        valid_values = [results[metric] for eps, results in eps_results.items() if results[metric] is not None]
        if valid_values:
            avg = sum(valid_values) / len(valid_values)
            avg_results[metric] = avg
            print(f"Average {metric} across all eps: {avg:.4f}")
            # 单个键值对传递平均值，键名格式为 metric_avg（如 acac_avg）
            ResultSender.send_result(f"{metric}_avg", f"{avg:.4f}")
        else:
            print(f"No valid values for {metric} across all eps")
            ResultSender.send_result(f"{metric}_avg", "None")

    return eps_results, avg_results


def evaluate_clean(test_loader, estimator):
    total_correct_clean = 0
    total_samples = 0

    for x_batch, y_batch in test_loader:
        x_batch_np = x_batch.numpy().astype(np.float32)
        y_batch_np = y_batch.numpy()
        bs = y_batch_np.shape[0]

        # 原始预测
        pred_clean = process_predictions(x_batch_np, estimator)

        # 对原始预测进行softmax归一化
        pred_clean_probs = softmax(pred_clean)
        total_correct_clean += np.sum(np.argmax(pred_clean_probs, axis=1) != y_batch_np)

        total_samples += bs

    # 计算整体准确率
    err_clean = 100 * total_correct_clean / total_samples
    print(f"asr_clean (full test set): {err_clean:.2f}%")
    return err_clean


def evaluate_robustness_corruptions(test_loader, estimator, metrics):
    ResultSender.send_log("进度", "扰动攻击评测开始")
    # 定义所有扰动方法
    corruption_functions = [
        gaussian_noise,
        # shot_noise, impulse_noise, speckle_noise,
        # gaussian_blur, glass_blur, defocus_blur, motion_blur, zoom_blur,
        # fog, 
        # frost, 
        # snow, spatter, contrast, brightness, saturate, jpeg_compression, 
        # pixelate, 
        # elastic_transform
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
                    if len(images.shape) == 4:
                        bs, c, h, w = images.shape
                    elif len(images.shape) == 5:
                        bs_crop, ncrops, c_crop, h_crop, w_crop = images.shape
                        bs = bs_crop  # 批次大小统一为第一维
                    else:
                        raise ValueError(f"不支持的图像维度: {images.shape}")

                    # 处理单图输入（4维）
                    if len(images.shape) == 4:
                        for i in range(bs):
                            # 转换为图像格式（HWC，0-255）
                            image = images[i].permute(1, 2, 0).numpy() * 255
                            image = image.astype(np.uint8)
                            # 应用扰动
                            corrupted_image = corruption_function(image, severity=severity)
                            # 转换回模型输入格式（CHW，0-1）
                            if isinstance(corrupted_image, np.ndarray):
                                corrupted_image = torch.from_numpy(corrupted_image / 255.0).permute(2, 0, 1).float()
                            else:
                                corrupted_image = transforms.ToTensor()(corrupted_image)
                            corrupted_image = corrupted_image.unsqueeze(0)  # 增加批次维度
                            
                            # 预测（单样本处理）
                            pred = process_predictions(corrupted_image.numpy(), estimator)
                            predicted = np.argmax(pred, axis=1)
                            total += 1
                            un_correct += (predicted[0] != labels[i].item())
                    # 处理10折裁剪输入（5维）
                    elif len(images.shape) == 5:
                        bs_crop, ncrops, c_crop, h_crop, w_crop = images.shape
                        for i in range(bs_crop):
                            # 取原始图像（非裁剪后）进行扰动
                            # 这里假设images[i,0]为原始图像，实际需根据数据处理逻辑调整
                            image = images[i, 0].permute(1, 2, 0).numpy() * 255
                            image = image.astype(np.uint8)
                            corrupted_image = corruption_function(image, severity=severity)
                            if isinstance(corrupted_image, np.ndarray):
                                corrupted_image = torch.from_numpy(corrupted_image / 255.0).permute(2, 0, 1).float()
                            else:
                                corrupted_image = transforms.ToTensor()(corrupted_image)
                            # 生成10个裁剪版本以匹配测试时的数据增强
                            corrupted_crops = transforms.TenCrop(44)(corrupted_image)
                            corrupted_crops = torch.stack([tc for tc in corrupted_crops]).unsqueeze(0)  # (1, 10, c, h, w)
                            
                            # 预测（处理5维数据）
                            pred = process_predictions(corrupted_crops.numpy(), estimator)
                            predicted = np.argmax(pred, axis=1)
                            total += 1
                            un_correct += (predicted[0] != labels[i].item())
                    else:
                        raise ValueError(f"不支持的图像维度: {images.shape}")
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
    if "mCE" in metrics:
        ResultSender.send_result("mCE", f"{(mCE / 100) :.4f}")
    if "RmCE" in metrics:
        err_clean = evaluate_clean(test_loader, estimator)
        RmCE = mCE - err_clean
        print(f"RmCE of the network on the test images: {RmCE:.2f}%")
        ResultSender.send_result("RmCE",  f"{(RmCE / 100) :.4f}")

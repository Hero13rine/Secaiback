# 加载数据
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt  # 添加matplotlib导入用于保存图像
import os  # 添加os模块用于处理文件路径

from attack import AttackFactory
from utils.SecAISender import ResultSender
from utils.visualize import denormalize  # 导入反归一化函数

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


def save_comparison_images(clean_img, adv_img, true_label, clean_pred, adv_pred, index, save_dir, eps=None):
    """保存原始图像和对抗样本的对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 显示原始图像
    if clean_img.shape[0] == 3:  # 如果是CHW格式
        clean_img_vis = denormalize(clean_img)
    else:  # 如果是HWC格式
        clean_img_vis = clean_img
    # 确保图像值在[0, 1]范围内
    clean_img_vis = np.clip(clean_img_vis, 0, 1)
    axes[0].imshow(clean_img_vis)
    axes[0].set_title(f"Clean Image\nTrue: {true_label}, Pred: {clean_pred}")
    axes[0].axis('off')

    # 显示对抗样本
    if adv_img.shape[0] == 3:  # 如果是CHW格式
        adv_img_vis = denormalize(adv_img)
    else:  # 如果是HWC格式
        adv_img_vis = adv_img
    # 确保图像值在[0, 1]范围内
    adv_img_vis = np.clip(adv_img_vis, 0, 1)
    axes[1].imshow(adv_img_vis)
    axes[1].set_title(f"Adversarial Image\nTrue: {true_label}, Pred: {adv_pred}")
    axes[1].axis('off')

    plt.tight_layout()
    filename = f"comparison_{index}.png"
    if eps is not None:
        filename = f"fgsm_eps_{str(eps).replace('.', '_')}_comparison_{index}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    return filename


def evaluate_robustness_adv(test_loader, estimator, attack, save_images=False, save_dir="adv_examples", eps=None):
    total_uncorrect_adv = 0
    total_samples = 0
    successful_attack_confidences = []  # 用于存储攻击成功样本的真实类别置信度
    acac_confidences = []  # 用于存储攻击成功样本的错误类别置信度

    # 用于保存图像的计数器
    saved_images_count = 0
    max_saved_images = 5  # 最多保存5组图像

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

        # 原始样本预测
        pred_clean = process_predictions(x_batch_np, estimator)
        pred_clean_probs = softmax(pred_clean)

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

            # 保存对比图像
            if save_images and saved_images_count < max_saved_images:
                # 获取预测标签
                clean_pred_label = np.argmax(pred_clean_probs[i])
                adv_pred_label = np.argmax(pred_adv_probs[i])

                # 只有当原始预测正确且对抗攻击成功时才保存图像
                if clean_pred_label == y_batch_np[i] and attack_success[i]:
                    filename = save_comparison_images(
                        x_batch_np[i], x_adv_np[i],
                        y_batch_np[i], clean_pred_label, adv_pred_label,
                        saved_images_count, save_dir, eps
                    )
                    saved_images_count += 1

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


def save_corruption_comparison(clean_img, corrupted_img, true_label, clean_pred, corrupted_pred, index, save_dir,
                               corruption_name, severity):
    """保存原始图像和扰动图像的对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 显示原始图像
    # 检查图像值范围来决定是否需要归一化
    if clean_img.max() > 1.0:
        clean_img_display = np.clip(clean_img / 255.0, 0, 1)
    else:
        clean_img_display = np.clip(clean_img, 0, 1)
    axes[0].imshow(clean_img_display)
    axes[0].set_title(f"Clean Image\nTrue: {true_label}, Pred: {clean_pred}")
    axes[0].axis('off')

    # 显示扰动后的图像
    # 检查图像值范围来决定是否需要归一化
    if corrupted_img.max() > 1.0:
        corrupted_img_display = np.clip(corrupted_img / 255.0, 0, 1)
    else:
        corrupted_img_display = np.clip(corrupted_img, 0, 1)
    axes[1].imshow(corrupted_img_display)
    axes[1].set_title(
        f"Corrupted Image\n{corruption_name} (severity={severity})\nTrue: {true_label}, Pred: {corrupted_pred}")
    axes[1].axis('off')

    plt.tight_layout()
    filename = f"comparison_{index}.png"
    filename = f"{corruption_name}_severity_{severity}_comparison_{index}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    return filename


def evaluate_robustness_adv_all(test_loader, estimator, metrics):
    ResultSender.send_log("进度", "对抗攻击评测开始")
    attack_method = ["fgsm"]

    # eps参数列表，从0到1，步长0.1
    eps_list = [round(eps, 1) for eps in np.arange(0.0, 1.1, 0.1)]

    # 存储所有eps的结果
    eps_results = {}  # 键是eps，值是包含各指标的字典

    # 选定两个eps值用于保存对比图
    selected_eps_for_saving = [0.3, 0.6] if len(eps_list) > 1 else [eps_list[0]]

    # 对每种攻击方法和eps值进行评估
    for attack_name in attack_method:
        for eps in eps_list:
            print(f"\nEvaluating {attack_name} with eps={eps}")
            attack_config = parse_attack_method(attack_name, eps)
            attack = AttackFactory.create(
                estimator=estimator.get_core(),
                config=attack_config
            )
            # 只在选定的eps值时保存图像
            save_images = eps in selected_eps_for_saving
            # 构建直接保存到结果目录的路径
            evaluateMetric = os.getenv("evaluateDimension")
            save_dir = None
            if save_images and evaluateMetric:
                save_dir = os.path.join("..", "evaluationData", evaluateMetric, "output")
                os.makedirs(save_dir, exist_ok=True)
            elif save_images:
                save_dir = f"adv_examples_{attack_name}_{str(eps).replace('.', '_')}"
                os.makedirs(save_dir, exist_ok=True)

            if save_dir:
                adverr, advacc, actc, acac = evaluate_robustness_adv(test_loader, estimator, attack,
                                                                     save_images=save_images,
                                                                     save_dir=save_dir, eps=eps)
            else:
                adverr, advacc, actc, acac = evaluate_robustness_adv(test_loader, estimator, attack,
                                                                     save_images=save_images)

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

        # 发送选定eps值的对比图路径
        try:
            # 获取环境变量
            evaluateMetric = os.getenv("evaluateDimension")
            resultPath = os.getenv("resultPath")

            if evaluateMetric and resultPath and selected_eps_for_saving:
                for eps in selected_eps_for_saving:
                    eps_str = str(eps).replace('.', '_')
                    # 直接在结果目录中查找图像
                    target_dir_rel = os.path.join("..", "evaluationData", evaluateMetric, "output")
                    target_dir_abs = os.path.join(resultPath, evaluateMetric, "output")

                    # 检查图像是否存在
                    target_img_name = f"fgsm_eps_{eps_str}_comparison_0.png"
                    target_img_path_rel = os.path.join(target_dir_rel, target_img_name)
                    target_img_path_abs = os.path.join(target_dir_abs, target_img_name)

                    print(f"检查对抗攻击图片路径: {target_img_path_rel}")
                    print(f"检查对抗攻击图片绝对路径: {target_img_path_abs}")

                    if os.path.exists(target_img_path_rel):
                        # 通过ResultSender发送路径
                        ResultSender.send_result(f"fgsm_eps_{eps_str}_comparison_0_path", target_img_path_abs)

                        # 打印保存路径
                        print(f"对抗攻击对比图已保存: {target_img_path_abs}")
                    else:
                        print(f"对抗攻击对比图不存在: {target_img_path_rel}")
            else:
                print("环境变量 evaluateDimension 或 resultPath 未设置，跳过发送对比图路径")
        except Exception as e:
            print(f"发送对抗攻击对比图路径时出错: {e}")

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

    # 选定两个severity级别用于保存对比图
    selected_severity_for_saving = [2, 4] if len(severity_levels) > 1 else [severity_levels[0]]

    # 遍历所有扰动方法
    for corruption_function in corruption_functions:
        for severity in severity_levels:
            total = 0
            un_correct = 0
            
            # 创建保存图像的目录
            save_dir = None
            # 只在选定的severity级别保存图像
            should_save_images = severity in selected_severity_for_saving
            if should_save_images:
                evaluateMetric = os.getenv("evaluateDimension")
                if evaluateMetric:
                    save_dir = os.path.join("..", "evaluationData", evaluateMetric, "output")
                    os.makedirs(save_dir, exist_ok=True)
                else:
                    save_dir = f"corruption_examples_{corruption_function.__name__}_{severity}"
                    os.makedirs(save_dir, exist_ok=True)

            # 用于保存图像的计数器
            saved_images_count = 0
            max_saved_images = 5  # 最多保存5组图像

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
                            
                            # 保存对比图像
                            if should_save_images and saved_images_count < max_saved_images:
                                # 原始图像预测
                                pred_clean = process_predictions(images[i].unsqueeze(0).numpy(), estimator)
                                clean_predicted = np.argmax(pred_clean, axis=1)

                                # 只有当原始预测正确且扰动后预测错误时才保存图像
                                if clean_predicted[0] == labels[i].item() and predicted[0] != labels[i].item():
                                    filename = save_corruption_comparison(
                                        image, corrupted_image.squeeze().permute(1, 2, 0).numpy() * 255,
                                        labels[i].item(), clean_predicted[0], predicted[0],
                                        saved_images_count, save_dir, corruption_function.__name__, severity
                                    )
                                    saved_images_count += 1
                            
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
                            
                            # 保存对比图像
                            if should_save_images and saved_images_count < max_saved_images:
                                # 原始图像预测
                                pred_clean = process_predictions(images[i, 0].unsqueeze(0).numpy(), estimator)
                                clean_predicted = np.argmax(pred_clean, axis=1)

                                # 只有当原始预测正确且扰动后预测错误时才保存图像
                                if clean_predicted[0] == labels[i].item() and predicted[0] != labels[i].item():
                                    filename = save_corruption_comparison(
                                        image, corrupted_image.permute(1, 2, 0).numpy() * 255,
                                        labels[i].item(), clean_predicted[0], predicted[0],
                                        saved_images_count, save_dir, corruption_function.__name__, severity
                                    )
                                    saved_images_count += 1
                            
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
            
            # 如果保存了图像，打印信息
            if should_save_images and saved_images_count > 0:
                print(
                    f"已保存 {saved_images_count} 组 {corruption_function.__name__} (severity={severity}) 对比图像到 {save_dir}")
            elif should_save_images:
                print(f"未找到合适的样本保存 {corruption_function.__name__} (severity={severity}) 对比图像")
            
            asr_total += err_corruption
    
        # 发送选定severity级别的对比图路径
        try:
            # 获取环境变量
            evaluateMetric = os.getenv("evaluateDimension")
            resultPath = os.getenv("resultPath")

            if evaluateMetric and resultPath and selected_severity_for_saving:
                for severity in selected_severity_for_saving:
                    # 直接在结果目录中查找图像
                    target_dir_rel = os.path.join("..", "evaluationData", evaluateMetric, "output")
                    target_dir_abs = os.path.join(resultPath, evaluateMetric, "output")

                    # 检查图像是否存在
                    target_img_name = f"{corruption_function.__name__}_severity_{severity}_comparison_0.png"
                    target_img_path_rel = os.path.join(target_dir_rel, target_img_name)
                    target_img_path_abs = os.path.join(target_dir_abs, target_img_name)

                    print(f"检查扰动攻击图片路径: {target_img_path_rel}")
                    print(f"检查扰动攻击图片绝对路径: {target_img_path_abs}")

                    if os.path.exists(target_img_path_rel):
                        # 通过ResultSender发送路径
                        ResultSender.send_result(
                            f"{corruption_function.__name__}_severity_{severity}_comparison_0_path",
                            target_img_path_abs)

                        # 打印保存路径
                        print(f"扰动攻击对比图已保存: {target_img_path_abs}")
                    else:
                        print(f"扰动攻击对比图不存在: {target_img_path_rel}")
            else:
                print("环境变量 evaluateDimension 或 resultPath 未设置，跳过发送对比图路径")
        except Exception as e:
            print(f"发送扰动攻击对比图路径时出错: {e}")

    mCE = asr_total / (len(corruption_functions) * len(severity_levels))
    print(f"mCE of the network on the test images: {mCE:.2f}%")
    if "mCE" in metrics:
        ResultSender.send_result("mCE", f"{(mCE / 100) :.4f}")
    if "RmCE" in metrics:
        err_clean = evaluate_clean(test_loader, estimator)
        RmCE = mCE - err_clean
        print(f"RmCE of the network on the test images: {RmCE:.2f}%")
        ResultSender.send_result("RmCE",  f"{(RmCE / 100) :.4f}")
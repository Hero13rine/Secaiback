# 加载数据
import numpy as np
import torch
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

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
        raise  # 保留异常抛出，方便调试

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
    if clean_img.shape[0] == 3:  # CHW格式
        clean_img_vis = denormalize(clean_img)
    else:  # HWC格式
        clean_img_vis = clean_img
    clean_img_vis = np.clip(clean_img_vis, 0, 1)
    axes[0].imshow(clean_img_vis)
    axes[0].set_title(f"Clean Image\nTrue: {true_label}, Pred: {clean_pred}")
    axes[0].axis('off')

    # 显示对抗样本
    if adv_img.shape[0] == 3:  # CHW格式
        adv_img_vis = denormalize(adv_img)
    else:  # HWC格式
        adv_img_vis = adv_img
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
    successful_attack_confidences = []
    acac_confidences = []

    saved_images_count = 0
    max_saved_images = 5

    for x_batch, y_batch in test_loader:
        x_batch_np = x_batch.numpy().astype(np.float32)
        y_batch_np = y_batch.numpy()
        bs = y_batch_np.shape[0]

        # 生成对抗样本
        if len(x_batch_np.shape) == 5:
            bs_adv, ncrops_adv, c_adv, h_adv, w_adv = x_batch_np.shape
            x_flat = x_batch_np.reshape(-1, c_adv, h_adv, w_adv)
            x_adv_flat = attack.generate(x_flat)
            x_adv_np = x_adv_flat.reshape(bs_adv, ncrops_adv, c_adv, h_adv, w_adv)
        else:
            x_adv_np = attack.generate(x_batch_np)

        # 对抗样本预测
        pred_adv = process_predictions(x_adv_np, estimator)
        pred_adv_probs = softmax(pred_adv)
        total_uncorrect_adv += np.sum(np.argmax(pred_adv_probs, axis=1) != y_batch_np)

        # 原始样本预测
        pred_clean = process_predictions(x_batch_np, estimator)
        pred_clean_probs = softmax(pred_clean)

        # 统计攻击成功样本置信度
        attack_success = np.argmax(pred_adv_probs, axis=1) != y_batch_np
        for i in range(bs):
            if attack_success[i]:
                true_class_confidence = pred_adv_probs[i][y_batch_np[i]]
                successful_attack_confidences.append(true_class_confidence)
                misclassified_class = np.argmax(pred_adv_probs[i])
                misclassified_confidence = pred_adv_probs[i][misclassified_class]
                acac_confidences.append(misclassified_confidence)

            # 保存对比图像
            if save_images and saved_images_count < max_saved_images:
                clean_pred_label = np.argmax(pred_clean_probs[i])
                adv_pred_label = np.argmax(pred_adv_probs[i])
                if clean_pred_label == y_batch_np[i] and attack_success[i]:
                    # 处理5维数据时取第一个裁剪图用于可视化
                    clean_img = x_batch_np[i][0] if len(x_batch_np.shape) == 5 else x_batch_np[i]
                    adv_img = x_adv_np[i][0] if len(x_adv_np.shape) == 5 else x_adv_np[i]
                    filename = save_comparison_images(
                        clean_img, adv_img,
                        y_batch_np[i], clean_pred_label, adv_pred_label,
                        saved_images_count, save_dir, eps
                    )
                    saved_images_count += 1

        total_samples += bs

    adverr = total_uncorrect_adv / total_samples
    advacc = 1 - adverr
    print(f"Adversarial dataset accuracy (full test set): {advacc:.2%}")
    print(f"Adversarial dataset error (full test set): {adverr:.2%}")

    # 计算ACTC和ACAC
    actc = np.mean(successful_attack_confidences) if successful_attack_confidences else None
    acac = np.mean(acac_confidences) if acac_confidences else None
    if actc is not None:
        print(f"actc (Average Confidence of True Class): {actc:.4f}")
    else:
        print("No successful attacks found. actc cannot be calculated.")
    if acac is not None:
        print(f"acac (Average Confidence of Adversarial Class): {acac:.4f}")
    else:
        print("No successful attacks found. acac cannot be calculated.")

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

    # 显示原始图像（确保是HWC格式且值在0-1之间）
    clean_img_display = np.clip(clean_img / 255.0 if clean_img.max() > 1.0 else clean_img, 0, 1)
    axes[0].imshow(clean_img_display)
    axes[0].set_title(f"Clean Image\nTrue: {true_label}, Pred: {clean_pred}")
    axes[0].axis('off')

    # 显示扰动后的图像（确保是HWC格式且值在0-1之间）
    corrupted_img_display = np.clip(corrupted_img / 255.0 if corrupted_img.max() > 1.0 else corrupted_img, 0, 1)
    axes[1].imshow(corrupted_img_display)
    axes[1].set_title(
        f"Corrupted Image\n{corruption_name} (severity={severity})\nTrue: {true_label}, Pred: {corrupted_pred}")
    axes[1].axis('off')

    plt.tight_layout()
    filename = f"{corruption_name}_severity_{severity}_comparison_{index}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    return filename

def evaluate_robustness_adv_all(test_loader, estimator, metrics):
    ResultSender.send_log("进度", "对抗攻击评测开始")
    attack_method = ["fgsm"]
    eps_list = [round(eps, 1) for eps in np.arange(0.0, 1.1, 0.1)]
    eps_results = {}
    selected_eps_for_saving = [0.3, 0.6] if len(eps_list) > 1 else [eps_list[0]]

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

            eps_results[eps] = {
                'adverr': adverr,
                'advacc': advacc,
                'actc': actc,
                'acac': acac
            }

            # 发送指标结果
            for metric in metrics:
                value = eps_results[eps][metric]
                eps_str = str(eps).replace('.', '_')
                key = f"{metric}_{eps_str}"
                if value is not None:
                    ResultSender.send_result(key, f"{value:.4f}")
                else:
                    ResultSender.send_result(key, "None")

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

    # 发送平均指标
    for metric in metrics:
        valid_values = [results[metric] for eps, results in eps_results.items() if results[metric] is not None]
        if valid_values:
            avg = sum(valid_values) / len(valid_values)
            print(f"Average {metric} across all eps: {avg:.4f}")
            ResultSender.send_result(f"{metric}_avg", f"{avg:.4f}")
        else:
            print(f"No valid values for {metric} across all eps")
            ResultSender.send_result(f"{metric}_avg", "None")

    return eps_results, avg  # 修正返回值（原avg_results未定义，直接返回avg）

def evaluate_clean(test_loader, estimator):
    total_incorrect_clean = 0  # 修正变量名（原total_correct_clean语义矛盾）
    total_samples = 0

    for x_batch, y_batch in test_loader:
        x_batch_np = x_batch.numpy().astype(np.float32)
        y_batch_np = y_batch.numpy()
        bs = y_batch_np.shape[0]

        # 原始预测
        pred_clean = process_predictions(x_batch_np, estimator)
        pred_clean_probs = softmax(pred_clean)
        # 统计预测错误的样本数（原逻辑正确，修正变量名使其语义清晰）
        total_incorrect_clean += np.sum(np.argmax(pred_clean_probs, axis=1) != y_batch_np)
        total_samples += bs

    err_clean = 100 * total_incorrect_clean / total_samples
    print(f"asr_clean (full test set): {err_clean:.2f}%")
    return err_clean

def get_original_image(images, idx):
    """从4维或5维张量中提取原始图像（用于可视化）"""
    if len(images.shape) == 4:
        # (bs, c, h, w) → 取单个样本并转HWC格式（0-255）
        return images[idx].permute(1, 2, 0).numpy() * 255
    elif len(images.shape) == 5:
        # (bs, ncrops, c, h, w) → 取第一个裁剪图并转HWC格式（0-255）
        return images[idx, 0].permute(1, 2, 0).numpy() * 255
    else:
        raise ValueError(f"不支持的图像维度: {images.shape}")

def evaluate_robustness_corruptions(test_loader, estimator, metrics):
    ResultSender.send_log("进度", "扰动攻击评测开始")
    # 定义扰动方法（可根据需要解除注释扩展）
    corruption_functions = [
        gaussian_noise,
        # shot_noise, impulse_noise, speckle_noise,
        # gaussian_blur, glass_blur, defocus_blur, motion_blur, zoom_blur,
        # fog, frost, snow, spatter, contrast, brightness, saturate,
        # jpeg_compression, pixelate, elastic_transform
    ]
    severity_levels = [1, 2, 3, 4, 5]
    asr_total = 0
    selected_severity_for_saving = [2, 4] if len(severity_levels) > 1 else [severity_levels[0]]

    for corruption_function in corruption_functions:
        corruption_name = corruption_function.__name__
        for severity in severity_levels:
            total_samples = 0
            incorrect_count = 0
            save_dir = None
            should_save_images = severity in selected_severity_for_saving

            # 创建保存目录
            if should_save_images:
                evaluateMetric = os.getenv("evaluateDimension")
                if evaluateMetric:
                    save_dir = os.path.join("..", "evaluationData", evaluateMetric, "output")
                    os.makedirs(save_dir, exist_ok=True)
                else:
                    save_dir = f"corruption_examples_{corruption_function.__name__}_{severity}"
                    os.makedirs(save_dir, exist_ok=True)

            saved_images_count = 0
            max_saved_images = 5

            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    bs = images.shape[0]  # 无论4维还是5维，批次大小都是第一维

                    for i in range(bs):
                        # 提取原始图像（用于扰动和可视化）
                        original_img = get_original_image(images, i).astype(np.uint8)
                        
                        # 应用扰动（输入必须是uint8格式的HWC图像）
                        corrupted_img = corruption_function(original_img, severity=severity)
                        
                        # 转换为模型输入格式：HWC → CHW，0-1归一化，添加批次维度
                        if isinstance(corrupted_img, np.ndarray):
                            # 处理numpy数组格式
                            corrupted_tensor = torch.from_numpy(corrupted_img / 255.0).permute(2, 0, 1).float()
                        else:
                            # 处理PIL图像格式
                            corrupted_tensor = transforms.ToTensor()(corrupted_img)
                        
                        # 根据输入数据类型生成对应格式的扰动数据
                        if len(images.shape) == 5:
                            # 5维数据：生成10折裁剪，保持格式为(1, 10, c, h, w)
                            ncrops = images.shape[1]
                            # TenCrop返回tuple，需转换为tensor并添加批次维度
                            corrupted_crops = transforms.TenCrop(size=original_img.shape[:2])(corrupted_tensor)
                            corrupted_crops = torch.stack(corrupted_crops).unsqueeze(0)  # (1, 10, c, h, w)
                            model_input = corrupted_crops.numpy()
                        else:
                            # 4维数据：保持格式为(1, c, h, w)
                            model_input = corrupted_tensor.unsqueeze(0).numpy()
                        
                        # 模型预测
                        pred = process_predictions(model_input, estimator)
                        pred_label = np.argmax(pred, axis=1)[0]
                        true_label = labels[i].item()

                        # 统计错误数
                        if pred_label != true_label:
                            incorrect_count += 1
                        total_samples += 1

                        # 保存对比图像（仅当原始预测正确且扰动后预测错误时）
                        if should_save_images and saved_images_count < max_saved_images:
                            # 原始图像的预测（使用完整输入格式）
                            original_input = images[i:i+1].numpy()  # 保持原始维度（1, ncrops, c, h, w）或（1, c, h, w）
                            pred_clean = process_predictions(original_input, estimator)
                            clean_pred_label = np.argmax(pred_clean, axis=1)[0]
                            
                            if clean_pred_label == true_label and pred_label != true_label:
                                save_corruption_comparison(
                                    original_img,  # 原始图像（HWC, 0-255）
                                    corrupted_img,  # 扰动图像（HWC, 0-255）
                                    true_label,
                                    clean_pred_label,
                                    pred_label,
                                    saved_images_count,
                                    save_dir,
                                    corruption_name,
                                    severity
                                )
                                saved_images_count += 1

            # 计算ASR（攻击成功率=错误数/总样本数）
            if total_samples > 0:
                asr = 100 * incorrect_count / total_samples
                asr_total += asr
                # 日志输出
                ResultSender.send_log("进度",
                                      f"UnCorrectNum of {corruption_name}_severity_{severity}: {incorrect_count}")
                ResultSender.send_log("进度",
                                      f"ASR of {corruption_name}_severity_{severity}: {asr:.2f}%")
                print(f"UnCorrectNum of {corruption_name}_severity_{severity}: {incorrect_count}")
                print(f"ASR of {corruption_name}_severity_{severity}: {asr:.2f}%")
                
                # 图像保存日志
                if should_save_images:
                    if saved_images_count > 0:
                        print(f"已保存 {saved_images_count} 组 {corruption_name}_severity_{severity} 对比图到 {save_dir}")
                    else:
                        print(f"未找到符合条件的样本（原始预测正确+扰动预测错误），未保存 {corruption_name}_severity_{severity} 对比图")
            else:
                print(f"警告：{corruption_name}_severity_{severity} 未处理任何样本")

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

    # 计算mCE（平均 corruption error）
    num_corruptions = len(corruption_functions)
    num_severities = len(severity_levels)
    if num_corruptions > 0 and num_severities > 0:
        mCE = asr_total / (num_corruptions * num_severities)
        print(f"mCE (Average Corruption Error): {mCE:.2f}%")
        if "mCE" in metrics:
            ResultSender.send_result("mCE", f"{mCE / 100:.4f}")  # 转换为小数形式

        # 计算RmCE（相对 mCE = mCE - 干净样本错误率）
        if "RmCE" in metrics:
            err_clean = evaluate_clean(test_loader, estimator)
            RmCE = mCE - err_clean
            print(f"RmCE (Relative mCE): {RmCE:.2f}%")
            ResultSender.send_result("RmCE", f"{RmCE / 100:.4f}")  # 转换为小数形式
    else:
        print("警告：未计算mCE（无扰动方法或severity级别）")
        if "mCE" in metrics:
            ResultSender.send_result("mCE", "0.0000")
        if "RmCE" in metrics:
            ResultSender.send_result("RmCE", "0.0000")
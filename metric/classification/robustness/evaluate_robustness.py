import numpy as np
import torch
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from attack import AttackFactory
from utils.SecAISender import ResultSender
from utils.visualize import denormalize  # 仍保留原反归一化函数

from .corruptions import (
    gaussian_noise, shot_noise, impulse_noise, speckle_noise,
    gaussian_blur, glass_blur, defocus_blur, motion_blur, zoom_blur,
    fog, frost, snow, spatter, contrast, brightness, saturate,
    jpeg_compression, pixelate, elastic_transform
)

# ============================================================
# 🔧 新增：自动检测图像值域并正确反归一化显示
# ============================================================
def safe_to_display(img):
    """智能检测图像值域和格式，自动转换为0-1的HWC格式以便imshow"""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    elif img.ndim == 2:
        # 灰度图像（HWC单通道）
        pass

    # 自动检测值域
    if img.max() <= 1.0 and img.min() >= 0.0:
        # 已经是 [0,1]
        return np.clip(img, 0, 1)
    elif img.max() > 10:
        # 可能是 [0,255]
        return np.clip(img / 255.0, 0, 1)
    elif img.min() < 0:
        # 可能是标准化后的 [-2,2]
        try:
            img = denormalize(img)
            return np.clip(img, 0, 1)
        except Exception:
            return np.clip((img + 1) / 2, 0, 1)
    else:
        return np.clip(img, 0, 1)

# ============================================================
# Softmax 函数
# ============================================================
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ============================================================
# 主函数入口：鲁棒性评测
# ============================================================
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
        raise

# ============================================================
# 统一预测函数（兼容4D/5D）
# ============================================================
def process_predictions(images_np, estimator):
    if len(images_np.shape) == 5:
        bs, ncrops, c, h, w = images_np.shape
        images_flat = images_np.reshape(-1, c, h, w)
        outputs = estimator.predict(images_flat)
        outputs_avg = outputs.reshape(bs, ncrops, -1).mean(axis=1)
        return outputs_avg
    elif len(images_np.shape) == 4:
        return estimator.predict(images_np)
    else:
        raise ValueError(f"不支持的数据维度: {images_np.shape}")

# ============================================================
# 保存对抗样本对比图
# ============================================================
def save_comparison_images(clean_img, adv_img, true_label, clean_pred, adv_pred, index, save_dir, eps=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    clean_img_vis = safe_to_display(clean_img)
    adv_img_vis = safe_to_display(adv_img)

    # 自动判断是否为灰度图（单通道或三通道数值一致）
    is_clean_gray = clean_img_vis.ndim == 2 or (clean_img_vis.ndim == 3 and np.allclose(clean_img_vis[..., 0], clean_img_vis[..., 1]) and np.allclose(clean_img_vis[..., 0], clean_img_vis[..., 2]))
    is_adv_gray = adv_img_vis.ndim == 2 or (adv_img_vis.ndim == 3 and np.allclose(adv_img_vis[..., 0], adv_img_vis[..., 1]) and np.allclose(adv_img_vis[..., 0], adv_img_vis[..., 2]))

    axes[0].imshow(clean_img_vis, cmap='gray' if is_clean_gray else None)
    axes[0].set_title(f"Clean Image\nTrue: {true_label}, Pred: {clean_pred}")
    axes[0].axis('off')

    axes[1].imshow(adv_img_vis, cmap='gray' if is_adv_gray else None)
    axes[1].set_title(f"Adversarial Image\nTrue: {true_label}, Pred: {adv_pred}")
    axes[1].axis('off')

    plt.tight_layout()
    filename = f"comparison_{index}.png"
    if eps is not None:
        filename = f"fgsm_eps_{str(eps).replace('.', '_')}_comparison_{index}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    return filename

# ============================================================
# 对抗攻击评测核心函数
# ============================================================
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
                misclassified_confidence = np.max(pred_adv_probs[i])
                acac_confidences.append(misclassified_confidence)

            if save_images and saved_images_count < max_saved_images:
                clean_pred_label = np.argmax(pred_clean_probs[i])
                adv_pred_label = np.argmax(pred_adv_probs[i])
                if clean_pred_label == y_batch_np[i] and attack_success[i]:
                    # 固定取第一个裁剪图（crop_idx=0）
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
    print(f"Adversarial dataset accuracy: {advacc:.2%}")
    print(f"Adversarial dataset error: {adverr:.2%}")

    actc = np.mean(successful_attack_confidences) if successful_attack_confidences else None
    acac = np.mean(acac_confidences) if acac_confidences else None
    if actc is not None:
        print(f"actc: {actc:.4f}")
    else:
        print("No successful attacks found. actc cannot be calculated.")
    if acac is not None:
        print(f"acac: {acac:.4f}")
    else:
        print("No successful attacks found. acac cannot be calculated.")

    return adverr, advacc, actc, acac

# ============================================================
# 解析攻击参数
# ============================================================
def parse_attack_method(attack_str, eps):
    return {
        "method": attack_str,
        "parameters": {
            "eps": eps,
            "step_size": 0.005
        }
    }

# ============================================================
# 保存扰动对比图
# ============================================================
def save_corruption_comparison(clean_img, corrupted_img, true_label, clean_pred, corrupted_pred, index, save_dir,
                               corruption_name, severity):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    clean_img_display = safe_to_display(clean_img)
    corrupted_img_display = safe_to_display(corrupted_img)

    # 自动判断是否为灰度图（单通道或三通道数值一致）
    is_clean_gray = clean_img_display.ndim == 2 or (clean_img_display.ndim == 3 and np.allclose(clean_img_display[..., 0], clean_img_display[..., 1]) and np.allclose(clean_img_display[..., 0], clean_img_display[..., 2]))
    is_corrupted_gray = corrupted_img_display.ndim == 2 or (corrupted_img_display.ndim == 3 and np.allclose(corrupted_img_display[..., 0], corrupted_img_display[..., 1]) and np.allclose(corrupted_img_display[..., 0], corrupted_img_display[..., 2]))

    axes[0].imshow(clean_img_display, cmap='gray' if is_clean_gray else None)
    axes[0].set_title(f"Clean\nTrue: {true_label}, Pred: {clean_pred}")
    axes[0].axis('off')

    axes[1].imshow(corrupted_img_display, cmap='gray' if is_corrupted_gray else None)
    axes[1].set_title(f"{corruption_name}\nSeverity={severity}\nTrue: {true_label}, Pred: {corrupted_pred}")
    axes[1].axis('off')

    plt.tight_layout()
    filename = f"{corruption_name}_severity_{severity}_comparison_{index}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    return filename

def evaluate_robustness_adv_all(test_loader, estimator, metrics):
    ResultSender.send_log("进度", "对抗攻击评测开始")
    attack_method = ["fgsm"]
    eps_list = [round(eps, 3) for eps in np.arange(0, 0.101, 0.001)]
    eps_results = {}
    selected_eps_for_saving = [0.003, 0.006] if len(eps_list) > 1 else [eps_list[0]]

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

def get_original_image(images, idx, is_gray=False):
    """
    从4维或5维张量中提取原始图像（用于可视化）
    固定取第一个裁剪图（crop_idx=0）
    Args:
        images: 输入张量（4D: [bs, c, h, w] 或 5D: [bs, ncrops, c, h, w]）
        idx: 样本索引
        is_gray: 是否为灰度图像（三通道数值一致）
    Returns:
        原始图像（HWC格式，0-255 uint8）
    """
    if len(images.shape) == 5:
        # 5维数据：[bs, ncrops, c, h, w] → 固定取第一个裁剪块（crop_idx=0）
        img = images[idx, 0]  # [c, h, w]
    else:
        # 4维数据：[bs, c, h, w]
        img = images[idx]  # [c, h, w]
    
    # 转HWC格式
    img = img.permute(1, 2, 0).numpy()  # [h, w, c]
    
    # 0-255归一化
    img = (img - img.min()) / (img.max() - img.min()) * 255  # 确保值域正确映射
    img = img.astype(np.uint8)
    
    # 灰度图像（三通道数值一致）转单通道
    if is_gray and img.shape[-1] == 3:
        img = img[..., 0]  # 取任意一个通道即可（三通道数值一致）
    
    return img

def check_gray_image(img_tensor, tolerance=1e-3):
    """
    检查图像是否为灰度图（通过判断三通道数值是否相似）
    兼容：1通道灰度图、3通道灰度图（三通道数值一致）
    Args:
        img_tensor: 单张图像张量（CHW格式: [c, h, w]）
        tolerance: 数值相似度容差（默认1e-3，可调整）
    Returns:
        bool: 是否为灰度图像
    """
    c = img_tensor.shape[0]
    if c == 1:
        # 1通道直接判定为灰度图
        return True
    elif c == 3:
        # 3通道：判断三个通道的数值是否在容差范围内一致
        channel1 = img_tensor[0].cpu().numpy()
        channel2 = img_tensor[1].cpu().numpy()
        channel3 = img_tensor[2].cpu().numpy()
        
        # 计算通道间的最大差异
        max_diff12 = np.max(np.abs(channel1 - channel2))
        max_diff13 = np.max(np.abs(channel1 - channel3))
        
        return max_diff12 < tolerance and max_diff13 < tolerance
    else:
        # 其他通道数暂不支持，默认判定为彩色图
        return False

def apply_corruption_to_crop(crop_img, corruption_func, severity, is_gray):
    """
    对单个裁剪块应用扰动（保持灰度/彩色一致性）
    Args:
        crop_img: 单个裁剪块张量 [c, h, w]（CHW格式）
        corruption_func: 扰动函数
        severity: 扰动强度
        is_gray: 是否为灰度图像（三通道数值一致）
    Returns:
        扰动后的裁剪块张量 [c, h, w]（CHW格式）
    """
    # 转HWC格式并归一化到0-255 uint8
    crop_hwc = crop_img.permute(1, 2, 0).numpy()  # [h, w, c]
    crop_hwc = (crop_hwc - crop_hwc.min()) / (crop_hwc.max() - crop_hwc.min()) * 255
    crop_hwc = crop_hwc.astype(np.uint8)
    
    # 灰度图像处理（转为单通道避免扰动函数生成彩色）
    if is_gray:
        if crop_hwc.shape[-1] == 3:
            # 3通道灰度图 → 单通道
            crop_hwc = crop_hwc[..., 0]  # 取第一个通道（三通道数值一致）
    
    # 应用扰动
    corrupted_hwc = corruption_func(crop_hwc, severity=severity)
    
    # 恢复通道数（灰度→保持单通道或转为3通道，彩色保持3通道）
    if is_gray:
        if corrupted_hwc.ndim == 2:
            # 单通道 → 恢复为原始通道数（1或3）
            if crop_img.shape[0] == 1:
                corrupted_hwc = np.expand_dims(corrupted_hwc, axis=-1)  # [h, w, 1]
            else:  # 原始为3通道灰度图
                corrupted_hwc = np.repeat(np.expand_dims(corrupted_hwc, axis=-1), 3, axis=-1)  # [h, w, 3]
        elif corrupted_hwc.ndim == 3 and corrupted_hwc.shape[-1] == 3:
            # 部分扰动函数可能输出3通道，转灰度（取均值或任意通道）
            corrupted_hwc = np.expand_dims(np.mean(corrupted_hwc, axis=-1), axis=-1).astype(np.uint8)
            if crop_img.shape[0] == 3:
                corrupted_hwc = np.repeat(corrupted_hwc, 3, axis=-1)  # 恢复3通道
    
    # 归一化到0-1并转CHW格式
    corrupted_chw = torch.from_numpy(corrupted_hwc / 255.0).permute(2, 0, 1).float()
    
    return corrupted_chw

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
    selected_severity_for_saving = [1, 2] if len(severity_levels) > 1 else [severity_levels[0]]

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
                        true_label = labels[i].item()
                        original_input = images[i:i+1].numpy()  # 原始输入（保持4D/5D格式）
                        pred_clean = process_predictions(original_input, estimator)
                        clean_pred_label = np.argmax(pred_clean, axis=1)[0]

                        # 检测当前样本是否为灰度图（取第一个裁剪块进行检测）
                        if len(images.shape) == 5:
                            sample_img = images[i, 0]  # [c, h, w]（第一个裁剪块）
                        else:
                            sample_img = images[i]  # [c, h, w]
                        is_gray = check_gray_image(sample_img)

                        # 生成扰动输入（关键修复：复用原始裁剪块，固定取第一个裁剪图用于扰动）
                        if len(images.shape) == 5:
                            # 5D数据：[bs, ncrops, c, h, w] → 对每个裁剪块单独扰动（保持裁剪位置一致）
                            ncrops = images.shape[1]
                            corrupted_crops = []
                            for crop_idx in range(ncrops):
                                # 提取单个裁剪块
                                crop_img = images[i, crop_idx]  # [c, h, w]
                                # 应用扰动（保持灰度/彩色一致性）
                                corrupted_crop = apply_corruption_to_crop(
                                    crop_img, corruption_function, severity, is_gray
                                )
                                corrupted_crops.append(corrupted_crop)
                            # 重组为5D张量：[1, ncrops, c, h, w]
                            corrupted_tensor = torch.stack(corrupted_crops).unsqueeze(0)
                        else:
                            # 4D数据：[bs, c, h, w] → 直接扰动
                            img = images[i]  # [c, h, w]
                            corrupted_tensor = apply_corruption_to_crop(
                                img, corruption_function, severity, is_gray
                            ).unsqueeze(0)  # [1, c, h, w]

                        # 转numpy格式用于预测
                        model_input = corrupted_tensor.numpy()
                        pred = process_predictions(model_input, estimator)
                        pred_label = np.argmax(pred, axis=1)[0]

                        # 统计错误数
                        if pred_label != true_label:
                            incorrect_count += 1
                        total_samples += 1

                        # 保存对比图像（仅当原始预测正确且扰动后预测错误时）
                        if should_save_images and saved_images_count < max_saved_images:
                            if clean_pred_label == true_label and pred_label != true_label:
                                # 提取可视化用的原始图像和扰动图像（均取第一个裁剪块）
                                clean_img_vis = get_original_image(images, i, is_gray)
                                # 扰动图像取第一个裁剪块
                                if len(corrupted_tensor.shape) == 5:
                                    corrupted_img_vis = corrupted_tensor[0, 0].permute(1, 2, 0).numpy()
                                else:
                                    corrupted_img_vis = corrupted_tensor[0].permute(1, 2, 0).numpy()
                                # 转0-255 uint8
                                corrupted_img_vis = (corrupted_img_vis * 255).astype(np.uint8)
                                # 灰度图像处理（三通道→单通道）
                                if is_gray and corrupted_img_vis.ndim == 3 and corrupted_img_vis.shape[-1] == 3:
                                    corrupted_img_vis = corrupted_img_vis[..., 0]

                                save_corruption_comparison(
                                    clean_img_vis,
                                    corrupted_img_vis,
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
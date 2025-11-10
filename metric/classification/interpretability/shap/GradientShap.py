import os
import shap
import torch
import numpy as np
import matplotlib

from utils.SecAISender import ResultSender
from metric.classification.interpretability.shap.imagePlot import image_plot_no_orig_nobar

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process_cropped_data(images: torch.Tensor) -> list[torch.Tensor]:
    """
    处理单裁剪/10折裁剪数据，返回所有裁剪版本的列表
    :param images: 输入张量，可能是4维（单裁剪）或5维（10折裁剪）
    :return: 所有裁剪版本的列表（单裁剪时列表长度为1）
    """
    if len(images.shape) == 5:
        # 10折裁剪数据：拆分为10个裁剪版本，每个版本shape为(batch, C, H, W)
        crop_versions = [images[:, i, ...] for i in range(images.shape[1])]
        print(f"处理10折裁剪数据，拆分为{len(crop_versions)}个版本，每个版本shape: {crop_versions[0].shape}")
        return crop_versions
    elif len(images.shape) == 4:
        # 单裁剪数据：直接包装为列表返回
        print(f"处理单裁剪数据，shape: {images.shape}")
        return [images]
    else:
        raise ValueError(f"不支持的数据维度: {len(images.shape)}，仅支持4维（单裁剪）或5维（10折裁剪）")


def GradientShap(model, test_loader):
    try:
        # 强制使用CPU（避免CUDA与NumPy转换冲突）
        device = torch.device("cpu")
        model.to(device)
        model.eval()  # 设置模型为评估模式

        ResultSender.send_log("进度", "开始选择背景图像和要解释的图像")
        # 1. 准备背景图像（处理所有裁剪版本，保持Tensor格式）
        background_list = []
        num_required = 200
        it = iter(test_loader)
        while len(background_list) * test_loader.batch_size < num_required:
            images, _ = next(it)
            # 转移到CPU并处理裁剪数据
            images = images.to(device)
            images_processed_list = process_cropped_data(images)
            for img in images_processed_list:
                background_list.append(img)
        
        # 拼接所有背景图像并截取需要的数量（保持Tensor格式）
        background = torch.cat(background_list, dim=0)[:num_required].to(torch.float32)
        print(f"背景图像最终shape: {background.shape}")

        # 2. 获取待解释图像（处理所有裁剪版本，保持Tensor格式）
        test_images, test_labels = next(it)
        test_images = test_images.to(device)
        test_images_processed_list = process_cropped_data(test_images)
        # 截取前5张图像进行解释（每个裁剪版本都取前5张）
        test_images_list = [img[:5].to(torch.float32) for img in test_images_processed_list]
        true_labels = test_labels[:5].cpu().tolist()  # 标签转换为CPU列表
        print(f"待解释图像每个版本的shape: {test_images_list[0].shape}")
        ResultSender.send_log("进度", "图像选择完成")

        # 3. SHAP值计算（对每个裁剪版本分别计算，保持Tensor输入）
        explainer = shap.GradientExplainer(model, background)
        ResultSender.send_log("进度", "开始计算每个裁剪版本的shap值")
        
        shap_results_all_versions = []
        indexes_all_versions = []
        for img in test_images_list:
            # SHAP支持Tensor输入，直接传入计算
            shap_values, indexes = explainer.shap_values(img, ranked_outputs=3)
            # 将SHAP结果转换为NumPy数组（便于后续融合）
            if isinstance(shap_values, list):
                shap_values_np = [sv.cpu().numpy() if isinstance(sv, torch.Tensor) else sv for sv in shap_values]
            else:
                shap_values_np = shap_values.cpu().numpy() if isinstance(shap_values, torch.Tensor) else shap_values
            shap_results_all_versions.append(shap_values_np)
            
            # 处理预测类别索引
            indexes_np = indexes.cpu().numpy() if isinstance(indexes, torch.Tensor) else indexes
            indexes_all_versions.append(indexes_np)
        
        ResultSender.send_log("进度", "所有裁剪版本的shap值计算完成")

        # 4. 融合SHAP结果（以平均为例）
        # 合并所有版本的SHAP值并取平均
        shap_values_merged = np.mean(shap_results_all_versions, axis=0)
        # 合并所有版本的预测类别索引并取众数（更稳定）
        indexes_merged = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=indexes_all_versions).astype(int)
        print(f"融合后SHAP值shape: {np.array(shap_values_merged).shape}")
        print(f"融合后类别索引shape: {indexes_merged.shape}")

        # 5. 图像格式转换（适配SHAP可视化要求：H×W×C）
        # 处理SHAP值维度：(class_num, batch, C, H, W) -> (class_num, batch, H, W, C)
        if isinstance(shap_values_merged, list):
            shap_values = [sv.transpose(0, 2, 3, 1) for sv in shap_values_merged]
        else:
            shap_values = [shap_values_merged[..., i].transpose(0, 2, 3, 1) for i in range(shap_values_merged.shape[-1])]
        
        # 待解释图像格式转换：取第一个裁剪版本，Tensor->NumPy，(batch, C, H, W)->(batch, H, W, C)
        images_np = test_images_list[0].permute(0, 2, 3, 1).cpu().numpy()
        labels = [[f"Class: {label}" for label in sample] for sample in indexes_merged.tolist()]
        print(f"可视化图像shape: {images_np.shape}")
        print(f"融合后SHAP值处理后长度: {len(shap_values)}, 单个SHAP值shape: {shap_values[0].shape}")

        # 6. 路径准备和创建
        evaluateMetric = os.getenv("evaluateDimension")
        resultPath = os.getenv("resultPath") # 发往数据库的nfs路径

        # 检查必要的环境变量
        if not evaluateMetric or not resultPath:
            missing_vars = []
            if not evaluateMetric:
                missing_vars.append("evaluateDimension")
            if not resultPath:
                missing_vars.append("resultPath")
            raise ValueError(f"缺少必要的环境变量: {', '.join(missing_vars)}")

        # 创建输出目录
        output_dir_rel = os.path.join("..", "evaluationData", evaluateMetric, "output")
        os.makedirs(output_dir_rel, exist_ok=True)
        print(f"输出目录创建完成: {output_dir_rel}")

        result_list = []

        # 7. 遍历保存每张图的原图和SHAP图
        for i in range(len(images_np)):
            # 原图保存
            plt.figure(figsize=(6, 6))
            img = images_np[i]
            # 处理图像归一化（适配0-1显示范围）
            img = np.clip(img, 0, 1)  # 确保像素值在0-1之间
            if img.shape[2] == 1:
                # 单通道灰度图
                plt.imshow(img.squeeze(), cmap='gray')
            else:
                # 多通道彩色图（确保通道顺序正确）
                plt.imshow(img)
            plt.axis("off")
            plt.title(f"Original Image (True Label: {true_labels[i]})", fontsize=12)
            orig_path_rel = os.path.join(output_dir_rel, f"image_{i}_orig.png")
            plt.savefig(orig_path_rel, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"原图保存完成: {orig_path_rel}")

            # SHAP图保存
            plt.figure(figsize=(10, 6))
            image_plot_no_orig_nobar(
                [sv[i:i+1] for sv in shap_values],  # 每个类别对应的当前图像融合后SHAP值
                pixel_values=images_np[i:i+1],      # 当前图像像素值
                labels=[labels[i]]                   # 当前图像的融合后预测类别标签
            )
            shap_path_rel = os.path.join(output_dir_rel, f"image_{i}_shap.png")
            plt.savefig(shap_path_rel, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"SHAP图保存完成: {shap_path_rel}")

            # 添加结果路径（数据库存储的绝对路径）
            result_list.append({
                "origin": os.path.join(resultPath, evaluateMetric, "output", f"image_{i}_orig.png"),
                "shap": os.path.join(resultPath, evaluateMetric, "output", f"image_{i}_shap.png")
            })

        # 8. 返回结果
        ResultSender.send_result("shap", result_list)
        ResultSender.send_status("成功")
        ResultSender.send_log("进度", "所有图像保存完成，评测结果已写回数据库")
        print("多版本融合SHAP解释性评测完成！")

    except Exception as e:
        error_msg = f"多版本SHAP评测失败: {str(e)}"
        print(f"[ERROR] {error_msg}")
        ResultSender.send_log("错误", error_msg)
        ResultSender.send_status("失败")
        raise  # 重新抛出异常便于调试
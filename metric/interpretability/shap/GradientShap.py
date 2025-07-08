import os
import shap
import torch
import matplotlib

from utils.SecAISender import ResultSender
from metric.interpretability.shap.imagePlot import image_plot_no_orig_nobar

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def GradientShap(model, test_loader):
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        ResultSender.send_log("进度", "开始选择背景图像和要解释的图像")
        # 1. 准备背景图像
        background_list = []
        num_required = 200
        it = iter(test_loader)
        while len(background_list) * test_loader.batch_size < num_required:
            images, _ = next(it)
            background_list.append(images)
        background = torch.cat(background_list, dim=0)[:num_required].to(torch.float32).to(device)
        # 2. 获取待解释图像
        test_images, test_labels = next(it)
        test_images = test_images[:5].to(torch.float32).to(device)
        true_labels = test_labels[:5].tolist()
        ResultSender.send_log("进度", "图像选择完成")

        # 3. SHAP值计算
        explainer = shap.GradientExplainer(model, background)
        ResultSender.send_log("进度", "开始计算特征的shap值")
        shap_values, indexes = explainer.shap_values(test_images, ranked_outputs=3)
        ResultSender.send_log("进度", "shap值计算完成")

        # 4. 图像格式转换
        shap_values = [shap_values[..., i].transpose(0, 2, 3, 1) for i in range(shap_values.shape[-1])]
        images = test_images.permute(0, 2, 3, 1).cpu().numpy()
        labels = [[f"Class: {label}" for label in sample] for sample in indexes.tolist()]

        # 5. 路径准备
        evaluateMetric = os.getenv("evaluateMetric")
        resultPath = os.getenv("resultPath") # 发往数据库的nfs路径

        result_list = []

        # 6. 遍历保存每张图的原图和SHAP图
        for i in range(len(images)):
            # 原图保存
            plt.figure()
            img = images[i]
            if img.shape[2] == 1:
                plt.imshow(img.squeeze(), cmap='gray')
            else:
                plt.imshow(img)
            plt.axis("off")
            plt.title(f"Original Image (Label: {true_labels[i]})")
            orig_path_rel = os.path.join("..", "evaluationData", evaluateMetric, "output", f"image_{i}_orig.png")
            plt.savefig(orig_path_rel, dpi=300, bbox_inches="tight")
            plt.close()

            # SHAP图保存
            plt.figure()
            image_plot_no_orig_nobar(
                [sv[i:i+1] for sv in shap_values],
                pixel_values=images[i:i+1],
                labels=[labels[i]]
            )
            shap_path_rel = os.path.join("..", "evaluationData", evaluateMetric, "output", f"image_{i}_shap.png")
            plt.savefig(shap_path_rel, dpi=300, bbox_inches="tight")
            plt.close()

            # 添加结果路径
            result_list.append({
                "origin": os.path.join(resultPath, evaluateMetric, "output", f"image_{i}_orig.png"),
                "shap": os.path.join(resultPath, evaluateMetric, "output", f"image_{i}_shap.png")
            })

        # 7. 返回结果
        ResultSender.send_result("shap", result_list)
        ResultSender.send_status("成功")
        ResultSender.send_log("进度", "所有图像保存完成，评测结果已写回数据库")

    except Exception as e:
        ResultSender.send_log("错误", str(e))

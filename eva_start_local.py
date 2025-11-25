import os
import sys

import numpy as np
import torch
from torch import optim

# from metric.classification.generalization.generalization import evaluate_generalization
# from metric.classification.interpretability.shap.GradientShap import GradientShap
# from metric.classification.safety.membershipinference.evaluate_mia import evaluate_mia
# from utils.SecAISender import ResultSender
# from metric.classification.robustness.evaluate_robustness import evaluation_robustness
# from metric.classification.fairness.fairness_metrics import calculate_fairness_metrics

from estimator import EstimatorFactory
from method import load_config
from model import load_model
# 修改导入语句，直接从 load_dataset 导入
from tests.fasterrcnn.load_dataset import load_data
from utils.sender import ConsoleResultSender as ResultSender
from utils.convert import convert_with_config
from metric.object_detection.generaliazation.generaliazation import evaluate_cross_dataset_generalization

def main():
    user_id = "local_user"  # 本地调试时使用固定的用户ID
    model_id = "local_model"  # 本地调试时使用固定的模型ID
    evaluation_type = "generalization"  # 本地调试时使用固定的评测维度 
    evaluation_path = "/wkm/secai/secai-common/config/user/model_pytorch_det_fasterrcnn_fairness.yaml"  # 本地调试时使用本地配置文件路径

    # 1.加载配置文件
    user_config = load_config(evaluation_path)
    model_instantiation_config = user_config["model"]["instantiation"]
    model_estimator_config = user_config["model"]["estimator"]
    evaluation_config = user_config["evaluation"]
    task = model_estimator_config.get("task", "classification")
    ResultSender.send_log("进度", "配置文件已加载完毕")

    # 2.初始化模型
    model = load_model(model_instantiation_config["model_path"], model_instantiation_config["model_name"]
                       , model_instantiation_config["weight_path"], model_instantiation_config["parameters"])
    ResultSender.send_log("进度", "模型初始化完成")

    # 3.获取优化器和损失函数

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)
    loss = None


    # 4.生成估计器
    estimator = EstimatorFactory.create(
        model=model,
        loss=loss,
        optimizer=optimizer,
        config=model_estimator_config
    )
    ResultSender.send_log("进度", "估计器已生成")

    # 5.加载数据
    test_loader = load_data("/wkm/data/dior/test/test")
    ResultSender.send_log("进度", "数据集已加载")

    # 6.根据传入的评测类型进行评测
    from metric.object_detection.basic.basic import cal_basic
    from metric.object_detection.fairness import evaluate_fairness_detection as calculate_fairness_metrics
    if evaluation_type == "basic":
        cal_basic(estimator, test_loader, evaluation_config["basic"])
    # elif evaluation_type == "robustness":
    #     evaluation_robustness(test_loader, estimator, evaluation_config["robustness"])
    # elif evaluation_type == "interpretability":
    #     GradientShap(model, test_loader)
    elif evaluation_type == "generalization":
        if task == "detection":
            convert_cfg = {
                "src_images": os.getenv("CONVERT_SRC_IMAGES", "/wkm/data/dota/DOTA/test/images"),
                "src_labels": os.getenv("CONVERT_SRC_LABELS", "/wkm/data/dota/DOTA/test/labels"),
                "dst_images": os.getenv("CONVERT_DST_IMAGES", "/wkm/data/dota_dior/test"),
                "dst_labels": os.getenv("CONVERT_DST_LABELS", "/wkm/data/dota_dior/test"),
                "src_classes": os.getenv("CONVERT_SRC_CLASSES", "/wkm/data/dota.txt"),
                "dst_classes": os.getenv("CONVERT_DST_CLASSES", "/wkm/data/dior.txt"),
            }
            if all(convert_cfg.values()):
                try:
                    ResultSender.send_log("进度", "开始转换 DOTA 标签")
                    convert_with_config(convert_cfg)
                    ResultSender.send_log("进度", "DOTA 标签转换完成")
                except Exception as exc:
                    ResultSender.send_log("警告", f"转换 DOTA 标签失败: {exc}")
            else:
                ResultSender.send_log("提示", "未配置 DOTA 转换路径，跳过标签转换")
        dataset_loaders = {
            "source_train": load_data("/wkm/data/dior/test/test"),
            "target_val": load_data("/wkm/data/dota_dior/test"),
        }
        evaluate_cross_dataset_generalization(estimator, dataset_loaders, evaluation_config["generalization"].get("generalization_testing"))    
    elif evaluation_type == "fairness":
        calculate_fairness_metrics(estimator, test_loader, evaluation_config["fairness"])
    ResultSender.send_log("进度", "评测结束")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ResultSender.send_log("错误", str(e))
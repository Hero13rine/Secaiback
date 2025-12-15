import os
import sys

import numpy as np
import torch
from torch import optim

from metric.classification.generalization.generalization import evaluate_generalization
from metric.classification.interpretability.shap.GradientShap import GradientShap
from metric.classification.safety.membershipinference.evaluate_mia import evaluate_mia
from utils.SecAISender import ResultSender
from metric.classification.robustness.evaluate_robustness import evaluation_robustness
from metric.classification.fairness.fairness_metrics import calculate_fairness_metrics
from metric.object_detection.robustness import (
    evaluation_robustness as detection_evaluation_robustness,
)
from metric.object_detection.generaliazation.generaliazation import (
    evaluate_cross_dataset_generalization,
)
from utils.convert import convert_with_config
# from update_table import update

from estimator import EstimatorFactory
from utils import load_config
from model import load_model
# 修改导入语句，直接从 load_dataset 导入
from utils.load_dataloader import load_dataloader


def main():
    # 将目标路径添加到系统路径
    sys.path.append('/app/userData/modelData/')
    sys.path.append('/app/systemData/database_code/')
    # 0.0获取当前 Pod 名称
    pod_name = os.getenv('HOSTNAME')  # 获取 Pod 名称（例如: 1242343443-1880539772613976065-basic）

    # 0.1从 Pod 名称中提取信息
    parts = pod_name.split('-')  # 根据 '-' 分割名称
    user_id = parts[0]  # 第一部分是用户ID
    model_id = parts[1]  # 第二部分是模型ID
    evaluation_type = parts[2]  # 第三部分是评测维度，如"basic"，"robustness"
    evaluation_path = "/app/userData/modelData/evaluationConfigs/" + "evaluationConfig" + ".yaml"

    # 1.加载配置文件
    user_config = load_config(evaluation_path)
    model_instantiation_config = user_config["model"]["instantiation"]
    model_estimator_config = user_config["model"]["estimator"]
    evaluation_config = user_config["evaluation"]
    task = model_estimator_config.get("task", "classification")
    ResultSender.send_log("进度", "配置文件已加载完毕")

    # 2.初始化模型
    model = load_model(
        model_instantiation_config["model_path"],
        model_instantiation_config["model_name"],
        model_instantiation_config["weight_path"],
        model_instantiation_config["parameters"]
    )
    ResultSender.send_log("进度", "模型初始化完成")

    # 3.获取优化器和损失函数
    if task == "detection":
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)
        loss = None
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss = torch.nn.CrossEntropyLoss()

    # 4.生成估计器
    estimator = EstimatorFactory.create(
        model=model,
        loss=loss,
        optimizer=optimizer,
        config=model_estimator_config,
    )
    ResultSender.send_log("进度", "估计器已生成")

    # 5.加载数据
    load_dataset = load_dataloader("./load_dataset.py")
    if task == "detection":
        if evaluation_type == "safety":
            train_loader, val_loader, test_loader = load_dataset()
        else:
            _, _, test_loader = load_dataset()
    else:
        if evaluation_type == "safety":
            train_loader, test_loader = load_dataset()
        else:
            _, test_loader = load_dataset()
    ResultSender.send_log("进度", "数据集已加载")

    # 6.根据传入的评测类型进行评测
    # 基础维度
    if task == "detection":
        from metric.object_detection.basic.basic import cal_basic
    else:
        from metric.classification.basic.basic import cal_basic
    if evaluation_type == "basic":
        cal_basic(estimator, test_loader, evaluation_config["basic"])
    # 鲁棒性维度
    elif evaluation_type == "robustness":
        if task == "detection":
            detection_evaluation_robustness(
                estimator=estimator,
                test_data=test_loader,
                config=evaluation_config,
                batch_size=64,
            )
        else:
            from metric.classification.robustness.evaluate_robustness import evaluation_robustness
            evaluation_robustness(test_loader, estimator, evaluation_config["robustness"])
    #可解释性维度
    elif evaluation_type == "interpretability":
        if task == "detection":
            from metric.object_detection.interpretability.fidelity import run_detection_interpretability
            run_detection_interpretability(
                model,
                estimator,
                test_loader,
                evaluation_config=evaluation_config["interpretability"],
            )
        else:
            GradientShap(model, test_loader)
    #泛化性维度
    elif evaluation_type == "generalization":
        if task == "detection":
            convert_cfg = {
                "src_images": os.getenv("CONVERT_SRC_IMAGES", "/app/systemData/evaluation_data/DOTA/test/images"),
                "src_labels": os.getenv("CONVERT_SRC_LABELS", "/app/systemData/evaluation_data/DOTA/test/labels"),
                "dst_images": os.getenv("CONVERT_DST_IMAGES", "/app/userData/modelData/data/dataset/dota/test"),
                "dst_labels": os.getenv("CONVERT_DST_LABELS", "/app/userData/modelData/data/dataset/dota/test"),
                "src_classes": os.getenv("CONVERT_SRC_CLASSES", "/app/systemData/evaluation_data/DOTA/test/dota.txt"),
                "dst_classes": os.getenv("CONVERT_DST_CLASSES", "/app/userData/modelData/classes.txt"),
            }
            if all(convert_cfg.values()):
                try:
                    ResultSender.send_log("进度", "开始转换数据集标签")
                    convert_with_config(convert_cfg)
                    ResultSender.send_log("进度", "数据集标签转换完成")
                except Exception as exc:
                    ResultSender.send_log("警告", f"转换数据集标签失败: {exc}")
            # 检测泛化评测
            _, _, source_loader = load_dataset(test_root="/app/userData/modelData/data/dataset/dior/test",
                                                                    batch_size=2,
                                                                    num_workers=0,
                                                                    augment_train=False,)
            _, _, target_loader = load_dataset(test_root="/app/userData/modelData/data/dataset/dota/test",
                                                                  batch_size=2,
                                                                  num_workers=0, )
            dataset_loaders = {
                "source_train": source_loader,
                "target_test": target_loader,
            }
            evaluate_cross_dataset_generalization(
                estimator,
                dataset_loaders,
                evaluation_config["generalization"]["generalization_testing"],
            )
        else:
            # 分类泛化评测
            evaluate_generalization(test_loader, estimator, evaluation_config["generalization"]["generalization_testing"])
    #安全性维度
    elif evaluation_type == "safety":
        if task == "detection":
            from metric.object_detection.safety.mia.evaluate_mia import evaluation_mia_detection as evaluate_mia

            evaluate_mia(train_loader, val_loader, test_loader, evaluation_config["safety"], model,
                                     model_instantiation_config)
        else:
            from metric.classification.safety.membershipinference.evaluate_mia import evaluate_mia
            evaluate_mia(train_loader, test_loader, estimator, evaluation_config["safety"]["membership_inference"])
    #公平性维度
    elif evaluation_type == "fairness":
            if task == "detection":
                from metric.object_detection.fairness import evaluate_fairness_detection
                evaluate_fairness_detection(estimator, test_loader, evaluation_config["fairness"])
            else:
                from metric.classification.fairness.fairness_metrics import calculate_fairness_metrics
                def sensitive_attribute_fn(images):
                # 示例：假设敏感属性是图像的奇偶索引
                    return np.array([i % 2 for i in range(len(images))])
                calculate_fairness_metrics(estimator, test_loader, sensitive_attribute_fn, evaluation_config["fairness"])

    ResultSender.send_log("进度", "评测结束")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ResultSender.send_log("错误", str(e))
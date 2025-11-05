import numpy as np
from sklearn.metrics import confusion_matrix
from utils.SecAISender import ResultSender


def calculate_fairness_metrics(estimator, test_loader, sensitive_attribute_fn, fairness_config):
    """
    计算公平性指标
    :param estimator: 模型估计器
    :param test_loader: 测试数据加载器
    :param sensitive_attribute_fn: 函数，用于提取敏感属性值
    :param metrics: 要计算的公平性指标列表
    """
    try:
        all_preds = []
        all_labels = []
        all_sensitive_attrs = []

        # 收集预测结果、真实标签和敏感属性
        for images, labels in test_loader:
            images_np = images.numpy()
            labels_np = labels.numpy()
            sensitive_attrs = sensitive_attribute_fn(images_np)

            outputs = estimator.predict(images_np)
            preds = np.argmax(outputs, axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_sensitive_attrs.extend(sensitive_attrs)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_sensitive_attrs = np.array(all_sensitive_attrs)

        results = {}

        # 从配置中提取所有要计算的公平性指标
        metrics = []
        if "group_fairness" in fairness_config:
            metrics.extend(fairness_config["group_fairness"])
        if "individual_fairness" in fairness_config:
            metrics.extend(fairness_config["individual_fairness"])

        # 计算SPD
        if "spd" in metrics:
            spd = statistical_parity_difference(all_preds, all_sensitive_attrs)
            results["spd"] = spd
            ResultSender.send_result("spd", spd)

        # 计算DIR
        if "dir" in metrics:
            dir = disparate_impact_ratio(all_preds, all_sensitive_attrs)
            results["dir"] = dir
            ResultSender.send_result("dir", dir)

        # 计算EOD
        if "eod" in metrics:
            eod = equal_opportunity_difference(all_preds, all_labels, all_sensitive_attrs)
            results["eod"] = eod
            ResultSender.send_result("eod", eod)

        # 计算AOD
        if "aod" in metrics:
            aod = average_odds_difference(all_preds, all_labels, all_sensitive_attrs)
            results["aod"] = aod
            ResultSender.send_result("aod", aod)

        # 计算一致性得分
        if "consistency" in metrics:
            consistency = consistency_score(all_preds)
            results["consistency"] = consistency
            ResultSender.send_result("consistency", consistency)

        ResultSender.send_status("成功")
        ResultSender.send_log("进度", "公平性评测结果已写回数据库")
        return results

    except Exception as e:
        print(f"公平性评测失败: {e}")
        ResultSender.send_status("失败")
        ResultSender.send_log("错误", str(e))


def statistical_parity_difference(preds, sensitive_attrs):
    group_0 = preds[sensitive_attrs == 0]
    group_1 = preds[sensitive_attrs == 1]
    return np.mean(group_0) - np.mean(group_1)


def disparate_impact_ratio(preds, sensitive_attrs):
    group_0 = preds[sensitive_attrs == 0]
    group_1 = preds[sensitive_attrs == 1]
    return np.mean(group_1) / np.mean(group_0)


def equal_opportunity_difference(preds, labels, sensitive_attrs):
    pos_mask = labels == 1  # 仅关注真实正类样本
    group_0 = (sensitive_attrs == 0) & pos_mask
    group_1 = (sensitive_attrs == 1) & pos_mask
    tpr_0 = np.mean(preds[group_0] == labels[group_0]) if np.sum(group_0) > 0 else 0
    tpr_1 = np.mean(preds[group_1] == labels[group_1]) if np.sum(group_1) > 0 else 0
    return tpr_1 - tpr_0


def average_odds_difference(preds, labels, sensitive_attrs):
    group_0 = sensitive_attrs == 0
    group_1 = sensitive_attrs == 1
    cm_0 = confusion_matrix(labels[group_0], preds[group_0])
    cm_1 = confusion_matrix(labels[group_1], preds[group_1])
    fpr_0 = cm_0[0, 1] / (cm_0[0, 1] + cm_0[0, 0])
    fpr_1 = cm_1[0, 1] / (cm_1[0, 1] + cm_1[0, 0])
    tpr_0 = cm_0[1, 1] / (cm_0[1, 1] + cm_0[1, 0])
    tpr_1 = cm_1[1, 1] / (cm_1[1, 1] + cm_1[1, 0])
    return 0.5 * ((fpr_1 - fpr_0) + (tpr_1 - tpr_0))


def consistency_score(preds):
    """
    计算相邻样本的预测一致性（适用于奇偶索引场景）
    """
    # 计算偶数索引与下一个奇数索引的一致性
    even_indices = np.arange(0, len(preds)-1, 2)
    odd_indices = even_indices + 1
    return np.mean(preds[even_indices] == preds[odd_indices])


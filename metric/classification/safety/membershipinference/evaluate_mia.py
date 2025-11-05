import numpy as np
import torch
from attack import AttackFactory
from sklearn.metrics import roc_curve, auc, average_precision_score
import logging
from typing import List, Tuple, Dict, Any, Optional
from utils.SecAISender import ResultSender  # 引入结果发送工具
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)


def extract_features_and_labels(
        data_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_inputs_list = []
    labels_list = []
    class_labels_list = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs_np = inputs.cpu().numpy()
            raw_inputs_list.append(inputs_np)

            labels_np = labels.cpu().numpy()
            labels_list.append(labels_np)
            class_labels_list.append(labels_np)

    return (
        np.vstack(raw_inputs_list),
        np.hstack(labels_list),
        np.hstack(class_labels_list)
    )


def generate_attack_data(
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        sample_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_member, y_member, class_member = extract_features_and_labels(train_loader)
    x_non_member, y_non_member, class_non_member = extract_features_and_labels(test_loader)

    if sample_size is not None:
        if len(x_member) > sample_size:
            indices = np.random.choice(len(x_member), sample_size, replace=False)
            x_member, y_member, class_member = x_member[indices], y_member[indices], class_member[indices]
        if len(x_non_member) > sample_size:
            indices = np.random.choice(len(x_non_member), sample_size, replace=False)
            x_non_member, y_non_member, class_non_member = x_non_member[indices], y_non_member[indices], \
                class_non_member[indices]

    return x_member, y_member, class_member, x_non_member, y_non_member, class_non_member


def evaluate_mia(
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        estimator: Any,
        safety_metrics: List[str],
        sample_size: Optional[int] = None,
):
    try:
        # 发送进度日志
        ResultSender.send_log("进度", "开始生成成员推理攻击数据")
        print("生成成员推理攻击数据中...")
        # 生成攻击数据
        x_member, y_member, class_member, x_non_member, y_non_member, class_non_member = generate_attack_data(
            train_loader, test_loader, sample_size
        )

        # 真实成员标签(1=成员, 0=非成员)
        y_true = np.hstack([np.ones(len(x_member)), np.zeros(len(x_non_member))])
        x_combined = np.vstack([x_member, x_non_member])
        y_combined = np.hstack([y_member, y_non_member])

        # 创建攻击对象
        internal_attack_config = {
            "method": "mia",
            "parameters": {"attack_model_type": "nn"}
        }
        ResultSender.send_log("进度", "创建成员推理攻击对象")
        print("创建成员推理攻击对象中...")
        attack = AttackFactory.create(estimator=estimator.get_core(), config=internal_attack_config)

        # 训练攻击模型
        ResultSender.send_log("进度", "训练成员推理攻击模型")
        print("训练成员推理攻击模型中...")
        attack.train(x_member, y_member, x_non_member, y_non_member)

        # 执行攻击
        ResultSender.send_log("进度", "执行成员推理攻击并获取预测概率")
        print("执行成员推理攻击中...")
        y_proba = attack.infer(x_combined, y_combined, probabilities=True)

        # 计算ROC曲线数据
        fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_true, y_proba)

        # 计算安全指标并发送结果
        ResultSender.send_log("进度", f"开始计算安全指标: {safety_metrics}")
        print("开始计算安全指标中...")
        metrics = {}

        if 'auc' in safety_metrics:
            auc_score = auc(fpr_roc, tpr_roc)
            metrics['auc'] = auc_score
            ResultSender.send_result("auc", auc_score)

        if 'attack_average_precision' in safety_metrics:
            ap_score = average_precision_score(y_true, y_proba)
            metrics['attack_average_precision'] = ap_score
            ResultSender.send_result("attack_average_precision", ap_score)

        # TPR@FPR=0.1%
        if 'tpr_at_fpr' in safety_metrics:
            fpr_target = 0.001  # 0.1%转换为小数
            valid_indices = np.where(fpr_roc <= fpr_target)[0]

            if len(valid_indices) == 0:
                idx = np.argmin(fpr_roc)
                logger.warning(f"目标FPR {fpr_target} 无法满足，使用最小FPR {fpr_roc[idx]:.6f}")
            else:
                idx = valid_indices[-1]

            tpr_value = tpr_roc[idx]
            threshold_value = thresholds_roc[idx]
            actual_fpr = fpr_roc[idx]

            metrics['tpr_at_fpr'] = tpr_value
            metrics['threshold'] = threshold_value
            metrics['actual_fpr'] = actual_fpr

            ResultSender.send_result("tpr_at_fpr", tpr_value)
            ResultSender.send_result("threshold", threshold_value)
            ResultSender.send_result("actual_fpr", actual_fpr)

        if 'roc_curve' in safety_metrics:
            try:
                # 获取环境变量
                evaluateMetric = os.getenv("evaluateMetric")
                resultPath = os.getenv("resultPath")

                # 添加环境变量检查
                if not evaluateMetric or not resultPath:
                    missing_vars = []
                    if not evaluateMetric: missing_vars.append("evaluateMetric")
                    if not resultPath: missing_vars.append("resultPath")
                    raise ValueError(f"缺少必要的环境变量: {', '.join(missing_vars)}")

                # 输出环境变量信息
                print(f"[DEBUG] evaluateMetric = {evaluateMetric}")
                print(f"[DEBUG] resultPath = {resultPath}")

                # 本地保存路径
                roc_filename_rel = os.path.join("..", "evaluationData", evaluateMetric, "output",
                                                "membership_inference_roc_curve.png")
                # 数据库记录路径
                roc_filename_abs = os.path.join(resultPath, evaluateMetric, "output",
                                                "membership_inference_roc_curve.png")

                # 输出路径信息
                print(f"[DEBUG] 本地相对路径: {roc_filename_rel}")
                print(f"[DEBUG] 数据库绝对路径: {roc_filename_abs}")

                # 创建目录
                output_dir = os.path.dirname(roc_filename_rel)
                print(f"[DEBUG] 创建输出目录: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
                print(f"[INFO] 目录创建成功: {os.path.exists(output_dir)}")

                # 绘制ROC曲线
                print("[DEBUG] 开始绘制ROC曲线...")
                plt.figure(figsize=(10, 8))
                plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=2,
                         label=f'ROC curve (AUC = {metrics.get("auc", 0):.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=14)
                plt.ylabel('True Positive Rate', fontsize=14)
                plt.title('Membership Inference Attack ROC Curve', fontsize=16)
                plt.legend(loc="lower right", fontsize=12)
                plt.grid(True, alpha=0.3)

                # 保存图片
                print(f"[DEBUG] 保存图片到: {roc_filename_rel}")
                plt.savefig(roc_filename_rel, dpi=300, bbox_inches='tight')
                plt.close()

                # 检查文件是否成功保存
                if os.path.exists(roc_filename_rel):
                    file_size = os.path.getsize(roc_filename_rel)
                    print(f"[INFO] ROC曲线保存成功! 文件大小: {file_size / 1024:.2f} KB")
                else:
                    raise FileNotFoundError(f"ROC曲线文件未生成: {roc_filename_rel}")

                # 发送结果
                metrics['roc_curve_path'] = roc_filename_abs
                print(f"[DEBUG] 发送结果路径到数据库: {roc_filename_abs}")
                ResultSender.send_result("roc_curve_path", roc_filename_abs)
                ResultSender.send_log("进度", "ROC曲线图像已保存并发送路径信息")
                print("[INFO] ROC曲线结果已发送")

            except Exception as roc_error:
                # 捕获并记录ROC生成过程中的具体错误
                error_msg = f"ROC曲线生成失败: {str(roc_error)}"
                print(f"[ERROR] {error_msg}")
                ResultSender.send_log("错误", error_msg)
                # 继续抛出异常以便外层捕获
                raise roc_error

        # 所有指标计算完成
        ResultSender.send_status("成功")
        ResultSender.send_log("进度", "安全指标计算完成，结果已存储")

    except Exception as e:
        # 发送错误信息
        ResultSender.send_status("失败")
        ResultSender.send_log("错误", f"安全指标计算失败: {str(e)}")
        raise  # 重新抛出异常便于调试
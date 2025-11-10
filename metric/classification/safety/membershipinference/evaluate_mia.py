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


def extract_features_and_labels_per_version(
        data_loader: torch.utils.data.DataLoader,
        target_num_versions: Optional[int] = None,  # 目标版本数（仅训练集使用，适配测试集）
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    提取每个裁剪版本的特征和标签（支持训练集适配测试集版本数）
    :param target_num_versions: 目标版本数（训练集用，如测试集是10版本则设为10）
    :return: (所有版本的特征列表, 标签, 类别标签)
             特征列表中每个元素shape: (total_samples, C, H, W)
    """
    version_features_list = []  # 存储每个裁剪版本的特征
    labels_list = []
    class_labels_list = []
    num_versions = None  # 原始裁剪版本数量（10或1）

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs_np = inputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # 判断是否为10折裁剪数据
            if len(inputs_np.shape) == 5:
                # 10折裁剪数据：(batch, 10, C, H, W)
                num_versions = inputs_np.shape[1]
                # 拆分每个版本，单独存储
                batch_version_features = [inputs_np[:, i, ...] for i in range(num_versions)]
            elif len(inputs_np.shape) == 4:
                # 单裁剪数据：包装为列表（版本数=1）
                num_versions = 1
                batch_version_features = [inputs_np]
            else:
                raise ValueError(f"不支持的数据维度: {len(inputs_np.shape)}， expected 4 (单裁剪) 或 5 (10折裁剪)")
            
            # 训练集适配目标版本数（如测试集10版本，训练集1版本→复制10份）
            if target_num_versions is not None and num_versions < target_num_versions:
                # 复制现有版本到目标数量（保持数据一致性）
                batch_version_features = batch_version_features * target_num_versions
                num_versions = target_num_versions
                logger.debug(f"训练集适配目标版本数{target_num_versions}，复制后版本数: {num_versions}")
            
            # 初始化版本特征列表（按版本索引存储）
            if not version_features_list:
                version_features_list = [[] for _ in range(num_versions)]
            
            # 按版本追加特征
            for i in range(num_versions):
                version_features_list[i].append(batch_version_features[i])
            
            labels_list.append(labels_np)
            class_labels_list.append(labels_np)

    # 拼接每个版本的特征（每个版本单独拼接为完整特征）
    final_version_features = []
    for i in range(num_versions):
        version_feat = np.concatenate(version_features_list[i], axis=0)
        final_version_features.append(version_feat)
        logger.debug(f"版本{i}特征shape: {version_feat.shape}")
    
    return (
        final_version_features,
        np.hstack(labels_list),
        np.hstack(class_labels_list)
    )


def generate_attack_data_per_version(
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        sample_size: Optional[int] = None
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """
    生成每个裁剪版本的攻击数据（自动适配训练集和测试集版本数）
    :return: (成员每个版本特征列表, 成员标签, 成员类别标签,
              非成员每个版本特征列表, 非成员标签, 非成员类别标签)
    """
    # 第一步：先获取测试集（非成员）的版本数（确定目标版本数）
    logger.debug("先获取测试集版本数...")
    test_version_feats_temp, _, _ = extract_features_and_labels_per_version(test_loader)
    target_num_versions = len(test_version_feats_temp)
    print(f"测试集（非成员）版本数: {target_num_versions}，训练集将适配此版本数")

    # 第二步：提取训练集（成员）数据（适配测试集版本数）
    logger.debug(f"提取训练集数据并适配版本数{target_num_versions}...")
    member_version_feats, y_member, class_member = extract_features_and_labels_per_version(
        train_loader, target_num_versions=target_num_versions
    )

    # 第三步：重新提取测试集（非成员）数据（避免临时变量占用内存）
    logger.debug("提取测试集（非成员）完整数据...")
    non_member_version_feats, y_non_member, class_non_member = extract_features_and_labels_per_version(
        test_loader, target_num_versions=None  # 测试集不修改版本数
    )

    # 校验成员和非成员的版本数量一致
    assert len(member_version_feats) == len(non_member_version_feats), \
        f"成员版本数({len(member_version_feats)})与非成员版本数({len(non_member_version_feats)})不匹配"
    num_versions = len(member_version_feats)
    print(f"最终版本数: {num_versions}（训练集已适配测试集）")

    # 采样（每个版本按相同索引采样，保证数据一致性）
    if sample_size is not None:
        # 成员数据采样
        if len(member_version_feats[0]) > sample_size:
            indices = np.random.choice(len(member_version_feats[0]), sample_size, replace=False)
            y_member = y_member[indices]
            class_member = class_member[indices]
            # 每个版本按相同索引采样
            member_version_feats = [feat[indices] for feat in member_version_feats]
        
        # 非成员数据采样
        if len(non_member_version_feats[0]) > sample_size:
            indices = np.random.choice(len(non_member_version_feats[0]), sample_size, replace=False)
            y_non_member = y_non_member[indices]
            class_non_member = class_non_member[indices]
            # 每个版本按相同索引采样
            non_member_version_feats = [feat[indices] for feat in non_member_version_feats]

    # 打印每个版本的维度信息
    for i in range(num_versions):
        print(f"版本{i} - 成员数据shape: {member_version_feats[i].shape}, 非成员数据shape: {non_member_version_feats[i].shape}")

    return (
        member_version_feats, y_member, class_member,
        non_member_version_feats, y_non_member, class_non_member
    )


def merge_mia_predictions(proba_list: List[np.ndarray], merge_strategy: str = "max_abs") -> np.ndarray:
    """
    融合多个版本的MIA预测概率
    :param proba_list: 每个版本的预测概率列表（shape: (n_samples,)）
    :param merge_strategy: 融合策略：max_abs（最大绝对值）、mean（平均，备用）、vote（投票）
    :return: 融合后的预测概率（shape: (n_samples,)）
    """
    proba_stack = np.stack(proba_list, axis=1)  # (n_samples, n_versions)
    
    if merge_strategy == "max_abs":
        # 最大绝对值融合（保留每个样本最显著的预测概率）
        abs_proba = np.abs(proba_stack)
        max_indices = np.argmax(abs_proba, axis=1)
        merged_proba = proba_stack[np.arange(len(proba_stack)), max_indices]
    elif merge_strategy == "vote":
        # 投票融合（概率>0.5视为正例，取多数投票结果）
        binary_preds = (proba_stack > 0.5).astype(int)
        merged_proba = np.mean(binary_preds, axis=1)  # 投票结果转为概率（0-1）
    elif merge_strategy == "mean":
        # 平均融合（备用）
        merged_proba = np.mean(proba_stack, axis=1)
    else:
        raise ValueError(f"不支持的融合策略: {merge_strategy}，可选：max_abs, vote, mean")
    
    return merged_proba


def evaluate_mia(
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        estimator: Any,
        safety_metrics: List[str],
        sample_size: Optional[int] = None,
        mia_merge_strategy: str = "max_abs"  # MIA预测融合策略
):
    try:
        # 发送进度日志
        ResultSender.send_log("进度", "成员推理攻击评测开始")
        ResultSender.send_log("进度", "开始生成每个裁剪版本的攻击数据")
        print("生成每个裁剪版本的攻击数据中...")
        
        # 生成每个版本的攻击数据（自动适配训练集和测试集版本数）
        (member_version_feats, y_member, class_member,
         non_member_version_feats, y_non_member, class_non_member) = generate_attack_data_per_version(
            train_loader, test_loader, sample_size
        )
        num_versions = len(member_version_feats)

        # 真实成员标签（1=成员, 0=非成员）
        y_true = np.hstack([np.ones(len(member_version_feats[0])), np.zeros(len(non_member_version_feats[0]))])

        # 创建攻击对象（复用一个攻击模型，按版本训练和预测）
        internal_attack_config = {
            "method": "mia",
            "parameters": {"attack_model_type": "nn"}
        }
        ResultSender.send_log("进度", "创建成员推理攻击对象")
        print("创建成员推理攻击对象中...")
        attack = AttackFactory.create(estimator=estimator.get_core(), config=internal_attack_config)

        # 每个版本单独训练并预测，收集所有版本的预测概率
        all_version_proba = []
        for i in range(num_versions):
            ResultSender.send_log("进度", f"训练并预测第{i}个裁剪版本的MIA模型")
            print(f"\n===== 处理第{i}个裁剪版本 =====")
            
            # 当前版本的训练数据
            x_member = member_version_feats[i]
            x_non_member = non_member_version_feats[i]
            
            # 合并当前版本的所有数据（用于预测）
            x_combined = np.concatenate([x_member, x_non_member], axis=0)
            y_combined = np.hstack([y_member, y_non_member])
            
            print(f"版本{i} - 训练数据shape: 成员{x_member.shape}, 非成员{x_non_member.shape}")
            print(f"版本{i} - 预测数据shape: {x_combined.shape}")

            # 训练攻击模型（每个版本单独训练）
            attack.train(x_member, y_member, x_non_member, y_non_member)

            # 执行攻击，获取预测概率
            y_proba = attack.infer(x_combined, y_combined, probabilities=True)
            
            # 确保预测结果是1维数组
            if len(y_proba.shape) > 1:
                y_proba = y_proba.squeeze()
            print(f"版本{i} - 预测概率shape: {y_proba.shape}")
            
            all_version_proba.append(y_proba)

        # 融合所有版本的预测概率（核心优化）
        ResultSender.send_log("进度", f"按{mia_merge_strategy}策略融合所有版本的MIA预测结果")
        print(f"\n按{mia_merge_strategy}策略融合{num_versions}个版本的预测结果...")
        y_proba_merged = merge_mia_predictions(all_version_proba, merge_strategy=mia_merge_strategy)
        print(f"融合后预测概率shape: {y_proba_merged.shape}")

        # 计算ROC曲线数据（基于融合后的概率）
        fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_true, y_proba_merged)

        # 计算安全指标并发送结果
        ResultSender.send_log("进度", f"开始计算安全指标: {safety_metrics}")
        print("开始计算安全指标中...")
        metrics = {}

        if 'auc' in safety_metrics:
            auc_score = auc(fpr_roc, tpr_roc)
            metrics['auc'] = auc_score
            ResultSender.send_result("auc", auc_score)
            print(f"AUC得分: {auc_score:.4f}")

        if 'attack_average_precision' in safety_metrics:
            ap_score = average_precision_score(y_true, y_proba_merged)
            metrics['attack_average_precision'] = ap_score
            ResultSender.send_result("attack_average_precision", ap_score)
            print(f"平均精度得分: {ap_score:.4f}")

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
            print(f"TPR@FPR=0.1%: {tpr_value:.4f}, 实际FPR: {actual_fpr:.6f}")

        if 'roc_curve' in safety_metrics:
            try:
                # 获取环境变量
                evaluateMetric = os.getenv("evaluateDimension")
                resultPath = os.getenv("resultPath")

                # 添加环境变量检查
                if not evaluateMetric or not resultPath:
                    missing_vars = []
                    if not evaluateMetric: missing_vars.append("evaluateDimension")
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

                # 绘制ROC曲线（基于融合后的结果）
                print("[DEBUG] 开始绘制ROC曲线...")
                plt.figure(figsize=(10, 8))
                plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=2,
                         label=f'ROC curve (AUC = {metrics.get("auc", 0):.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=14)
                plt.ylabel('True Positive Rate', fontsize=14)
                plt.title(f'Membership Inference Attack ROC Curve (Merge: {mia_merge_strategy})', fontsize=16)
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
        # 所有指标计算完成
        ResultSender.send_status("成功")
        ResultSender.send_log("进度", "安全指标计算完成，结果已存储")
    except Exception as e:
        # 捕获并记录外层评测流程中的异常，确保 try 有对应的 except
        error_msg = f"评测流程发生异常: {str(e)}"
        print(f"[ERROR] {error_msg}")
        ResultSender.send_log("错误", error_msg)
        ResultSender.send_status("失败")
        raise
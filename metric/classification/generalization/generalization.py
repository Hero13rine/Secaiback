import numpy as np
import torch


from utils.SecAISender import ResultSender  # 引入结果发送工具


# 保持原softmax函数不变
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def evaluate_generalization(test_loader, estimator, generalization_testing):
    """
    评估模型的泛化能力指标（无返回值，通过ResultSender发送结果）

    参数:
        test_loader: 测试数据集加载器
        estimator: 待评估的模型
        generalization_testing: 配置文件中"generalization.generalization_testing"对应的列表
    """
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 发送开始日志
    ResultSender.send_log("进度", "开始计算泛化能力指标")
    print(f"将计算的泛化指标: {generalization_testing}")

    # 存储所有批次的预测概率和分数
    all_pred_probs = []
    all_scores = []

    # 遍历测试集批次
    for x_batch, y_batch in test_loader:
        # 将数据移至CPU并转换为NumPy数组
        x_batch_np = x_batch.cpu().numpy().astype(np.float32)

        batch_size = x_batch_np.shape[0]

        # 直接判断数据格式并处理
        if len(x_batch_np.shape) == 5:  # 10折裁剪格式：(bs, 10, c, h, w)
            ResultSender.send_log("调试", f"检测到10折裁剪数据，批次shape: {x_batch_np.shape}")
            
            # 关键修复：reshape为4维输入（bs*10, c, h, w），适配Conv2d
            bs, ncrops, c, h, w = x_batch_np.shape
            x_reshaped = x_batch_np.reshape(-1, c, h, w)  # (128*10=1280, 3, 44, 44)
            
            # 模型预测（输出shape：(bs*10, num_classes)）
            pred = estimator.predict(x_reshaped)
            
            # 恢复为原shape：(bs, 10, num_classes)
            pred = pred.reshape(bs, ncrops, -1)
            
            # 步骤1：对每个样本的10个裁剪预测取平均概率（融合逻辑）
            # 先计算每个裁剪的softmax概率
            pred_probs_per_crop = softmax(pred.reshape(-1, pred.shape[-1]))  # (bs*10, num_classes)
            pred_probs_per_crop = pred_probs_per_crop.reshape(batch_size, 10, -1)  # 恢复为(bs, 10, num_classes)
            # 按样本维度平均，得到最终概率：(bs, num_classes)
            final_probs = pred_probs_per_crop.mean(axis=1)
            
            # 步骤2：计算每个样本的最终分数（融合后的最大logit或概率，保持与单张逻辑一致）
            # 方式：先取每个裁剪的最大logit，再平均；或直接用融合后概率的最大值（二选一，这里选前者更贴合原逻辑）
            max_logit_per_crop = np.max(pred, axis=-1)  # (bs, 10)
            final_scores = max_logit_per_crop.mean(axis=1)  # (bs,)

        elif len(x_batch_np.shape) == 4:  # 单张图片格式：(bs, c, h, w)
            ResultSender.send_log("调试", f"检测到单张图片数据，批次shape: {x_batch_np.shape}")
            
            # 模型预测（输出shape：(bs, num_classes)）
            pred = estimator.predict(x_batch_np)
            
            # 计算概率和分数（与原逻辑一致）
            final_probs = softmax(pred)  # (bs, num_classes)
            final_scores = np.max(pred, axis=1)  # (bs,)

        else:
            raise ValueError(f"未知的数据格式！输入shape: {x_batch_np.shape}，仅支持4维（单张）或5维（10折裁剪）")

        # 存储结果
        all_pred_probs.append(final_probs)
        all_scores.append(final_scores)

    # 合并所有批次的结果
    all_pred_probs = np.vstack(all_pred_probs)
    all_scores = np.hstack(all_scores)

    # 1. 计算平均MSP
    if "msp" in generalization_testing:
        msp_values = np.max(all_pred_probs, axis=1)
        avg_msp = np.mean(msp_values)
        ResultSender.send_result("msp", avg_msp)
        ResultSender.send_log("信息", f"平均MSP: {avg_msp:.4f}")

    # 2. 计算平均预测熵
    if "entropy" in generalization_testing:
        epsilon = 1e-12
        log_probs = np.log(all_pred_probs + epsilon)
        entropy_values = -np.sum(all_pred_probs * log_probs, axis=1)
        avg_entropy = np.mean(entropy_values)
        ResultSender.send_result("entropy", avg_entropy)
        ResultSender.send_log("信息", f"平均预测熵: {avg_entropy:.4f}")

    # 3. 计算Rademacher复杂度
    if "rademacher" in generalization_testing:
        n = len(all_scores)
        num_trials = 10  # 可从配置中读取
        rademacher_trials = []

        for _ in range(num_trials):
            sigma = np.random.choice([-1, 1], size=n)
            trial_val = np.mean(sigma * all_scores)
            rademacher_trials.append(trial_val)

        rademacher_complexity = float(np.mean(rademacher_trials))
        ResultSender.send_result("rademacher", rademacher_complexity)
        ResultSender.send_log("信息", f"Rademacher复杂度: {rademacher_complexity:.4f}")
        rademacher_trials = [float(x) for x in rademacher_trials]
        ResultSender.send_result("rademacher_trials", rademacher_trials)

    # 发送完成日志
    ResultSender.send_status("成功")
    ResultSender.send_log("进度", "泛化能力指标计算完成")
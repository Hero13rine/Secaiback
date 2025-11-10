import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix

from utils.SecAISender import ResultSender

def cal_basic(estimator, test_loader, metrics):
    estimator.get_core().model.eval()
    try:
        # 收集运行结果
        all_preds = []
        all_labels = []
        ResultSender.send_log("进度", "开始收集网络输出")
        for images, labels in test_loader:
            # 转换为numpy数组（保持原始维度信息）
            images_np = images.numpy()
            labels_np = labels.numpy()
            # 根据数据维度判断处理方式（4维单图/5维10折裁剪）
            if len(images_np.shape) == 5:  # (bs, ncrops, c, h, w)，对应10折裁剪数据
                bs, ncrops, c, h, w = images_np.shape
                # 展平裁剪维度：(bs*ncrops, c, h, w)
                images_flat = images_np.reshape(-1, c, h, w)
                # 模型预测
                outputs = estimator.predict(images_flat)
                # 按裁剪维度取平均：(bs, ncrops, num_classes) -> (bs, num_classes)
                outputs_avg = outputs.reshape(bs, ncrops, -1).mean(axis=1)
                # 计算预测结果
                preds = np.argmax(outputs_avg, axis=1)
            elif len(images_np.shape) == 4:  # (bs, c, h, w)，对应单图输入
                # 直接预测
                outputs = estimator.predict(images_np)
                preds = np.argmax(outputs, axis=1)
            else:
                raise ValueError(f"不支持的数据维度：{images_np.shape}，仅支持4维或5维")
            
            # 收集结果（注意labels需要展平为1维）
            all_preds.extend(preds)
            all_labels.extend(labels_np.flatten())  # 确保labels是1维列表
        print(all_preds, all_labels)

        ResultSender.send_log("进度", "网络输出收集完成")
        ResultSender.send_log("进度", "开始计算"+str(metrics.get('performance_testing', [])))
        metrics=metrics.get('performance_testing', [])

        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        # 计算基础指标
        if 'accuracy' in metrics:
            # 整体准确率
            accuracy = accuracy_score(all_labels, all_preds)
            ResultSender.send_result("accuracy", accuracy)
        if 'precision' in metrics:
            # 整体精确率
            precision = precision_score(all_labels, all_preds, average='macro')
            ResultSender.send_result("precision", precision)
            # 每个类别的精确率
            per_precision = {}
            for label, metrics in report.items():
                if label.isdigit():
                    per_precision[label] = metrics["precision"]
            ResultSender.send_result("per_precision", per_precision)
        if 'recall' in metrics:
            # 整体召回率
            recall = recall_score(all_labels, all_preds, average='macro')
            ResultSender.send_result("recall", recall)
            # 每个类别的召回率
            per_recall = {}
            for label, metrics in report.items():
                if label.isdigit():
                    per_recall[label] = metrics["recall"]
            ResultSender.send_result("per_recall", per_recall)
        if 'f1score' in metrics:
            # 整体f1score
            f1 = f1_score(all_labels, all_preds, average='macro')
            ResultSender.send_result("f1score", f1)
            # 每个类别的f1score
            per_f1score = {}
            for label, metrics in report.items():
                if label.isdigit():
                    per_f1score[label] = metrics["f1-score"]
            ResultSender.send_result("per_f1score", per_f1score)

        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        ResultSender.send_result("confusion_matrix", cm.tolist())

        ResultSender.send_status("成功")
        ResultSender.send_log("进度", "评测结果已写回数据库")

    except Exception as e:
        ResultSender.send_status("失败")
        ResultSender.send_log("错误", str(e))

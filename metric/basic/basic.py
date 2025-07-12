import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from utils.SecAISender import ResultSender

def cal_basic(estimator, test_loader, metrics):
    try:
        # 收集运行结果
        all_preds = []
        all_labels = []
        ResultSender.send_log("进度", "开始收集网络输出")
        for images, labels in test_loader:
            images_np = images.numpy()
            labels_np = labels.numpy()
            outputs = estimator.predict(images_np)
            preds = np.argmax(outputs, axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels)

        ResultSender.send_log("进度", "网络输出收集完成")
        ResultSender.send_log("进度", "开始计算"+metrics)

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

        ResultSender.send_status("成功")
        ResultSender.send_log("进度", "评测结果已写回数据库")

    except Exception as e:
        ResultSender.send_status("失败")
        ResultSender.send_log("错误", str(e))

import json
import os

import requests


class ResultSender:
    # 从环境变量中取出写入结果接口、结果类型、写入日志接口、模型Id、容器名称等
    modelId = os.getenv("modelId")
    containerName = os.getenv("containerName")
    evaluateDimension = os.getenv("evaluateDimension")
    evaluateMetric = os.getenv("evaluateMetric")
    evaluationType = os.getenv("evaluationType")
    logUrl = os.getenv("logUrl")
    resultUrl = os.getenv("resultUrl")
    resultColumn = os.getenv("resultColumn")
    statusUrl = resultUrl + "/status"

    @staticmethod
    def send_result(*args):
        """
        发送评测结果到 /result 接口
        :param args: 成对的键值参数，如 ("攻击成功率", "95%", "得分", 85)
        """
        # 将参数转换为字典
        result = {str(args[i]): str(args[i + 1]) if not isinstance(args[i + 1], dict)
        else json.dumps(args[i + 1]) for i in range(0, len(args), 2)}
        print("评测结果：", result)
        payload = {
            "modelId": int(ResultSender.modelId),
            "result": result,
            "resultColumn": ResultSender.resultColumn
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(ResultSender.resultUrl, json=payload, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            ResultSender.send_log("错误", str(e))
            print(f"评测结果发送失败: {e}")

    @staticmethod
    def send_log(message_key: str, message_value: str):
        """
        发送评测结果到 /log 接口
        :param message_key: 日志类型，message_value：日志值，如 ("进度", "50%")
        """
        data = {
            "modelId": ResultSender.modelId,
            "containerName": ResultSender.containerName,
            "evaluateDimension": ResultSender.evaluateDimension,
            "evaluateMetric": ResultSender.evaluateMetric,
            "messageKey": message_key,
            "messageValue": message_value
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(ResultSender.logUrl, json=data, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"日志发送失败: {e}")

    @staticmethod
    def send_status(status: str):
        """
        发送Pod状态到 /status 接口
        :param status: Pod状态， "成功"、"失败"
        """
        data = {
            "modelId": ResultSender.modelId,
            "metric": ResultSender.evaluationType,
            "status": status
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(ResultSender.statusUrl, json=data, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            ResultSender.send_log("错误", str(e))
            print(f"状态发送失败: {e}")


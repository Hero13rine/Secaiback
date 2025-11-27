"""SecAI 评测结果发送工具。

本地调试时，我们往往无法连接到生产环境提供的日志/结果接口，
导致看不到服务端收到的内容。为了在本地也能获得一致的输出，
本模块提供了**远程发送器**与**本地控制台发送器**两种实现。

生产环境保持默认导入 ``ResultSender``（远程实现）。本地调试只需
``from utils.SecAISender import ConsoleResultSender as ResultSender`` 即可。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import requests


def _pair_to_dict(*args: Any) -> Dict[str, str]:
    """将成对参数转换为字典，便于两个实现复用相同的序列化逻辑。"""

    if len(args) % 2 != 0:
        raise ValueError("参数必须成对提供，例如 ('key', 'value')")

    return {
        str(args[i]): json.dumps(args[i + 1], ensure_ascii=False)
        if isinstance(args[i + 1], (dict, list))
        else str(args[i + 1])
        for i in range(0, len(args), 2)
    }


class RemoteResultSender:
    """真实环境下，通过 HTTP 接口推送评测结果。"""

    # 从环境变量中取出写入结果接口、结果类型、写入日志接口、模型Id、容器名称等
    modelId = os.getenv("modelId")
    containerName = os.getenv("containerName")
    evaluateDimension = os.getenv("evaluateDimension")
    evaluateMetric = os.getenv("evaluateMetric")
    evaluationType = os.getenv("evaluationType")
    logUrl = os.getenv("logUrl")
    resultUrl = os.getenv("resultUrl")
    resultColumn = os.getenv("resultColumn")
    statusUrl = resultUrl + "/status" if resultUrl else None

    @staticmethod
    def send_result(*args: Any):
        """
        发送评测结果到 /result 接口。
        :param args: 成对的键值参数，如 ("攻击成功率", "95%", "得分", 85)
        """

        result = _pair_to_dict(*args)
        payload = {
            "modelId": int(RemoteResultSender.modelId),
            "result": result,
            "resultColumn": RemoteResultSender.resultColumn,
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(RemoteResultSender.resultUrl, json=payload, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            RemoteResultSender.send_log("错误", str(e))
            print(f"评测结果发送失败: {e}")

    @staticmethod
    def send_log(message_key: str, message_value: str):
        """
        发送评测结果到 /log 接口。
        :param message_key: 日志类型，message_value：日志值，如 ("进度", "50%")
        """
        data = {
            "modelId": RemoteResultSender.modelId,
            "containerName": RemoteResultSender.containerName,
            "evaluateDimension": RemoteResultSender.evaluateDimension,
            "evaluateMetric": RemoteResultSender.evaluateMetric,
            "messageKey": message_key,
            "messageValue": message_value,
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(RemoteResultSender.logUrl, json=data, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"日志发送失败: {e}")

    @staticmethod
    def send_status(status: str):
        """
        发送Pod状态到 /status 接口。
        :param status: Pod状态， "成功"、"失败"
        """
        data = {
            "modelId": RemoteResultSender.modelId,
            "metric": RemoteResultSender.evaluationType,
            "status": status,
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(RemoteResultSender.statusUrl, json=data, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            RemoteResultSender.send_log("错误", str(e))
            print(f"状态发送失败: {e}")


class ConsoleResultSender:
    """用于本地调试的发送器，仅在终端打印信息，不依赖任何远程接口。"""

    prefix = "[SecAI-Local]"

    @staticmethod
    def _model_id_as_int():
        return (
            int(RemoteResultSender.modelId)
            if RemoteResultSender.modelId is not None
            else None
        )

    @staticmethod
    def _result_payload(result: Dict[str, str]) -> Dict[str, Any]:
        return {
            "modelId": ConsoleResultSender._model_id_as_int(),
            "resultColumn": RemoteResultSender.resultColumn,
            "result": result,
        }

    @staticmethod
    def _log_payload(message_key: str, message_value: str) -> Dict[str, Any]:
        return {
            "modelId": RemoteResultSender.modelId,
            "containerName": RemoteResultSender.containerName,
            "evaluateDimension": RemoteResultSender.evaluateDimension,
            "evaluateMetric": RemoteResultSender.evaluateMetric,
            "messageKey": message_key,
            "messageValue": message_value,
        }

    @staticmethod
    def _status_payload(status: str) -> Dict[str, Any]:
        return {
            "modelId": RemoteResultSender.modelId,
            "metric": RemoteResultSender.evaluationType,
            "status": status,
        }

    @staticmethod
    def _print(message: str, payload: Dict[str, Any]):
        print(f"{ConsoleResultSender.prefix} {message}:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")

    @staticmethod
    def send_result(*args: Any):
        result = _pair_to_dict(*args)
        ConsoleResultSender._print("result", ConsoleResultSender._result_payload(result))

    @staticmethod
    def send_log(message_key: str, message_value: str):
        ConsoleResultSender._print("log", ConsoleResultSender._log_payload(message_key, message_value))

    @staticmethod
    def send_status(status: str):
        ConsoleResultSender._print("status", ConsoleResultSender._status_payload(status))


# 默认保持与历史行为一致：生产端导入 ResultSender 获得远程实现。
# 当缺少必要的远程地址（例如在本地调试或未注入环境变量时），
# 自动退回到控制台输出，避免出现 "Invalid URL 'None'" 的报错。
if RemoteResultSender.logUrl and RemoteResultSender.resultUrl:
    ResultSender = RemoteResultSender
else:
    ResultSender = ConsoleResultSender
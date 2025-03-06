import torch
from art.estimators.classification import PyTorchClassifier
from estimator.estimator_class.base_estimator import BaseEstimator
from estimator.estimator_factory import EstimatorFactory


@EstimatorFactory.register(framework="pytorch", task="classification")
class PyTorchClassificationWrapper(BaseEstimator):
    """智能适配参数的具体实现"""

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss=torch.nn.CrossEntropyLoss(),
                 device="auto",
                 input_shape=(3, 224, 224),
                 num_classes=10,
                 clip_values=(0, 1),
                 device_type="auto"):
        # 设备自动检测
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 设备自动检测
        if device_type == "auto":
            device_type = "gpu" if torch.cuda.is_available() else "cpu"

        # 初始化ART分类器
        self.core = PyTorchClassifier(
            model=model.to(device),
            loss=loss,
            optimizer=optimizer,
            clip_values=clip_values,
            input_shape=input_shape,
            device_type=device_type,
            nb_classes=num_classes
        )

    def predict(self, x):
        return self.core.predict(x)

    def get_core(self):
        return self.core

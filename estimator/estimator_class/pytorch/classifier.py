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
                 loss,
                 input_shape,
                 nb_classes,
                 clip_values,
                 use_amp,
                 opt_level,
                 loss_scale,
                 channels_first,
                 device_type,
                 device
                 ):
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
            input_shape=input_shape,
            nb_classes=nb_classes,
            clip_values=clip_values,
            use_amp=use_amp,
            opt_level=opt_level,
            loss_scale=loss_scale,
            channels_first=channels_first,
            device_type=device_type,
        )

    def predict(self, x):
        return self.core.predict(x)

    def get_core(self):
        return self.core

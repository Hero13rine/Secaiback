# model_definitions/fasterrcnn_factory.py
from typing import Optional
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class TorchvisionDetector:
    """
    工厂类：实例化时直接返回 torchvision 的检测模型实例。
    """

    def __init__(
        self,
        num_classes: int,
        network: str = "fasterrcnn",
        pretrained: bool = True,
        **kwargs,
    ):
        # 不做任何事，只是为了签名能被 inspect 识别
        pass

    def __new__(
        cls,
        num_classes: int,
        network: str = "fasterrcnn",
        pretrained: bool = True,
        **kwargs,
    ):
        number_classes = int(num_classes) + 1
        network = network.lower()

        if network != "fasterrcnn":
            raise ValueError(f"仅支持 'fasterrcnn'，收到: {network}")

        weights = "DEFAULT" if pretrained else None
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False,
        pretrained_backbone=False
)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, number_classes)

        return model


def custom_forward(model, images, targets):
    # Get model structure
    if model.training:
        return model(images, targets)
    else:
        return model(images)

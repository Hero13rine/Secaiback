"""
Wrapper to expose torchvision detection models for dynamic loading via load_model.py.
"""

import torchvision


def fasterrcnn_resnet50_fpn(pretrained: bool = True, num_classes: int = 91, **kwargs):
    """
    返回带 roi_heads 的 torchvision Faster R-CNN，兼容 Grad-CAM 的生成。
    默认为 COCO 预训练，按需调整 num_classes/kwargs。
    """
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )

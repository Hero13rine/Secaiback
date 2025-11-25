"""Default shadow model definition for MIA detection flow."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class ShadowFasterRCNN(nn.Module):
    """Faster R-CNN wrapper used for trained shadow weights.

    The class intentionally avoids downloading pretrained weights during loading
    because the trained weight file already contains tuned parameters.
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        self.model = model

    def forward(self, images, targets=None):  # type: ignore[override]
        return self.model(images, targets)

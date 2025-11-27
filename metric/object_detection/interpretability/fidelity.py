"""
Detection interpretability fidelity utilities.

We perturb each detection by keeping only the box region and by dropping it,
then compare confidence changes to estimate fidelity without relying on
task-specific Grad-CAM hooks.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

# guided_filter_pytorch is optional; skip guided filtering if not installed.
try:
    from guided_filter_pytorch.guided_filter import GuidedFilter
except ImportError:  # pragma: no cover - optional dependency
    GuidedFilter = None
from scipy.ndimage import gaussian_filter

# 远程/本地 Sender 自动切换：本地调试前需设置 `set LOCAL_SENDER=1`
_LOCAL_SENDER_ENABLED = bool(os.getenv("LOCAL_SENDER"))

if _LOCAL_SENDER_ENABLED:
    try:
        from utils.sender import ConsoleResultSender as ResultSender
    except ImportError as exc:
        raise RuntimeError(
            "LOCAL_SENDER=1 但未找到 utils/sender.ConsoleResultSender，"
            "请确认本地 sender.py 已放置或取消 LOCAL_SENDER 环境变量。"
        ) from exc
else:
    from utils.SecAISender import ResultSender as RemoteSender

    ResultSender = RemoteSender


@dataclass
class _FidelityRecord:
    """Internal record for fidelity scores."""

    original_score: float
    keep_ratio: float
    drop_ratio: float
    label: int
    image_index: int


@dataclass
class FidelitySummary:
    """Aggregated fidelity statistics."""

    samples: int
    keep_mean: float
    keep_std: float
    drop_mean: float
    drop_std: float


class DetectionFidelityMeter:
    """
    Compute perturbation-based fidelity for detector predictions.

    Args:
        estimator: estimator exposing ``predict`` for inference.
        score_threshold: minimum confidence for detections to consider.
        topk: maximum detections per image to evaluate.
        iou_match_threshold: IoU needed to match perturbed prediction back.
        perturbation: occlusion strategy, ``"zero"`` or ``"blur"``.
        keep_mask_outside: zero-out pixels outside box when keep_region=True.
    """

    def __init__(
        self,
        estimator,
        score_threshold: float = 0.25,
        topk: int = 5,
        iou_match_threshold: float = 0.5,
        perturbation: str = "blur",
        keep_mask_outside: bool = True,
    ) -> None:
        self.estimator = estimator
        self.score_threshold = score_threshold
        self.topk = topk
        self.iou_match_threshold = iou_match_threshold
        self.perturbation = perturbation
        self.keep_mask_outside = keep_mask_outside
        self._records: List[_FidelityRecord] = []

    def update_batch(
        self,
        images: Sequence[torch.Tensor],
        detections: Sequence[Dict[str, np.ndarray]],
        _: Optional[Sequence[Dict]] = None,
    ) -> None:
        """
        Update fidelity statistics with a mini-batch.

        Args:
            images: sequence of CHW tensors in [0,1].
            detections: predictions aligned with ``images`` containing
                ``boxes``, ``scores``, ``labels``.
            _: placeholder for targets (unused).
        """

        for index, (image, pred) in enumerate(zip(images, detections)):
            tensor = self._to_tensor(image)
            scores = pred.get("scores")
            boxes = pred.get("boxes")
            labels = pred.get("labels")
            if boxes is None or scores is None or labels is None:
                continue

            if len(boxes) == 0:
                continue

            order = np.argsort(scores)[::-1]
            kept = 0
            for idx in order:
                score = float(scores[idx])
                if score < self.score_threshold:
                    continue
                box = boxes[idx]
                label = int(labels[idx])

                keep_ratio, drop_ratio = self._evaluate_single_detection(
                    tensor, box, score, label
                )
                if keep_ratio is not None and drop_ratio is not None:
                    self._records.append(
                        _FidelityRecord(
                            original_score=score,
                            keep_ratio=keep_ratio,
                            drop_ratio=drop_ratio,
                            label=label,
                            image_index=index,
                        )
                    )
                    kept += 1
                if kept >= self.topk:
                    break

    def compute(self) -> Optional[FidelitySummary]:
        """Return aggregated fidelity statistics."""

        if not self._records:
            return None

        keep = np.array([rec.keep_ratio for rec in self._records], dtype=np.float32)
        drop = np.array([rec.drop_ratio for rec in self._records], dtype=np.float32)

        return FidelitySummary(
            samples=len(self._records),
            keep_mean=float(keep.mean()),
            keep_std=float(keep.std(ddof=0)),
            drop_mean=float(drop.mean()),
            drop_std=float(drop.std(ddof=0)),
        )

    # ------------------------------------------------------------------ #
    # 内部辅助方法
    # ------------------------------------------------------------------ #

    def _evaluate_single_detection(
        self,
        image: torch.Tensor,
        box: np.ndarray,
        original_score: float,
        label: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        with torch.no_grad():
            keep_img = self._mask_image(image, box, keep_region=True)
            drop_img = self._mask_image(image, box, keep_region=False)

            keep_score = self._query_score(keep_img, box, label)
            drop_score = self._query_score(drop_img, box, label)

        denom = max(original_score, 1e-6)
        keep_ratio = keep_score / denom
        drop_ratio = 1.0 - (drop_score / denom)

        return keep_ratio, drop_ratio

    def _query_score(
        self, image: torch.Tensor, ref_box: np.ndarray, label: int
    ) -> float:
        preds = self.estimator.predict([image])
        if not preds:
            return 0.0
        pred = preds[0]
        boxes = pred.get("boxes", np.zeros((0, 4), dtype=np.float32))
        scores = pred.get("scores", np.zeros((0,), dtype=np.float32))
        labels = pred.get("labels", np.zeros((0,), dtype=np.int64))

        if len(boxes) == 0:
            return 0.0

        best = 0.0
        for box, score, lab in zip(boxes, scores, labels):
            if int(lab) != label:
                continue
            iou = self._compute_iou(ref_box, box)
            if iou >= self.iou_match_threshold and score > best:
                best = float(score)
        return best

    def _mask_image(
        self,
        image: torch.Tensor,
        box: np.ndarray,
        keep_region: bool,
    ) -> torch.Tensor:
        tensor = image.clone()
        c, h, w = tensor.shape
        x1, y1, x2, y2 = self._sanitize_box(box, w, h)
        if x2 <= x1 or y2 <= y1:
            return tensor

        if keep_region:
            if self.keep_mask_outside:
                mask = torch.zeros_like(tensor)
                mask[:, y1:y2, x1:x2] = tensor[:, y1:y2, x1:x2]
                tensor = mask
        else:
            if self.perturbation == "zero":
                tensor[:, y1:y2, x1:x2] = 0.0
            elif self.perturbation == "blur":
                region = tensor[:, y1:y2, x1:x2]
                if region.numel() > 0:
                    mean_color = region.mean(dim=(1, 2), keepdim=True)
                    tensor[:, y1:y2, x1:x2] = mean_color
            else:
                raise ValueError(f"不支持的扰动方式: {self.perturbation}")
        return tensor

    @staticmethod
    def _sanitize_box(box: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = int(max(0, min(width - 1, np.floor(x1))))
        y1 = int(max(0, min(height - 1, np.floor(y1))))
        x2 = int(max(0, min(width, np.ceil(x2))))
        y2 = int(max(0, min(height, np.ceil(y2))))
        return x1, y1, x2, y2

    @staticmethod
    def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b

        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        denom = max(area_a + area_b - inter_area, 1e-6)
        return float(inter_area / denom)

    @staticmethod
    def _to_tensor(image: torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.detach().clone()
        else:
            raise TypeError("图像必须是torch.Tensor实例")
        if tensor.ndim != 3:
            raise ValueError("期望图像为CHW形状的张量")
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor.clamp(0.0, 1.0)


class GradCamGenerator:
    """Generate Grad-CAM overlays for torchvision-style detectors."""

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: Optional[Iterable[str]] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "outputs/grad_cam",
        alpha: float = 0.5,
        blur_sigma: float = 30.0,
        use_guided_filter: bool = True,
        gamma: float = 1.1,
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = device or next(self.model.parameters()).device
        self.class_names = list(class_names) if class_names else None
        self.output_dir = output_dir
        self.alpha = alpha
        self.blur_sigma = blur_sigma
        self.use_guided_filter = use_guided_filter and GuidedFilter is not None
        self.gamma = gamma
        self.roi_pool = getattr(self.model.roi_heads, "box_roi_pool", None)
        if self.roi_pool is None:
            raise ValueError("model.roi_heads.box_roi_pool is required for Grad-CAM generation")
        predictor = getattr(self.model.roi_heads, "box_predictor", None)
        cls_score = getattr(predictor, "cls_score", None) if predictor else None
        self.num_classes = getattr(cls_score, "out_features", None)
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(
        self,
        image_tensor: torch.Tensor,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: Optional[np.ndarray] = None,
        tag: Optional[str] = None,
    ) -> Optional[str]:
        if boxes is None or len(boxes) == 0:
            return None
        tensor = self._to_chw_tensor(image_tensor)
        rgb = self._tensor_to_rgb(tensor)
        cam = self._compute_roi_cam(tensor, boxes, labels, rgb)
        if cam is None:
            return None
        overlay = self._apply_heatmap(rgb, cam)
        overlay = self._draw_detections(overlay, boxes, labels, scores)
        filename = f"{tag or uuid.uuid4().hex}.jpg"
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return path

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _to_chw_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("Grad-CAM expects a torch.Tensor image")
        tensor = image_tensor.detach().clone()
        if tensor.ndim != 3:
            raise ValueError("Expected CHW tensor for Grad-CAM")
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor.to(self.device)

    @staticmethod
    def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
        array = tensor.detach().cpu().numpy()
        array = np.clip(array, 0.0, 1.0)
        array = np.transpose(array, (1, 2, 0))
        return (array * 255.0).astype(np.uint8)

    def _compute_roi_cam(
        self,
        image_tensor: torch.Tensor,
        boxes: np.ndarray,
        labels: np.ndarray,
        rgb_image: np.ndarray,
    ) -> Optional[np.ndarray]:
        h, w = image_tensor.shape[1:]
        torch_boxes = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)
        torch_labels = torch.as_tensor(labels, dtype=torch.int64, device=self.device)

        transformed = self.model.transform([image_tensor])
        images_list = transformed[0] if isinstance(transformed, tuple) else transformed
        features = self.model.backbone(images_list.tensors)
        proposals = [torch_boxes]

        activations = {}
        gradients = {}

        def fw_hook(_module, _inputs, output):
            activations["roi"] = output

        def bw_hook(_module, grad_inputs, grad_outputs):
            gradients["roi"] = grad_outputs[0]

        h_fw = self.roi_pool.register_forward_hook(fw_hook)
        h_bw = self.roi_pool.register_full_backward_hook(bw_hook)

        try:
            roi_feats = self.model.roi_heads.box_roi_pool(
                features, proposals, images_list.image_sizes
            )
            box_feats = self.model.roi_heads.box_head(roi_feats)
            class_logits, _ = self.model.roi_heads.box_predictor(box_feats)

            merged = np.zeros((h, w), dtype=np.float32)
            for idx in range(torch_boxes.shape[0]):
                cls_id = int(torch_labels[idx].item())
                if cls_id < 0:
                    continue
                if self.num_classes is not None and cls_id >= self.num_classes:
                    continue

                self.model.zero_grad(set_to_none=True)
                class_logits[idx, cls_id].backward(retain_graph=True)

                roi_act = activations.get("roi")
                roi_grad = gradients.get("roi")
                if roi_act is None or roi_grad is None:
                    continue

                feat = roi_act[idx]
                grad = roi_grad[idx]
                weights = grad.mean(dim=(1, 2), keepdim=True)
                cam = torch.relu((weights * feat).sum(dim=0)).detach().cpu().numpy()
                cam = self._normalize(cam)
                merged = np.maximum(
                    merged,
                    self._paste_cam(cam, boxes[idx], h, w),
                )
        finally:
            h_fw.remove()
            h_bw.remove()

        if self.use_guided_filter:
            merged = self._guided_filter(rgb_image, merged)
        return self._normalize(merged)

    def _paste_cam(self, cam: np.ndarray, box: np.ndarray, h: int, w: int) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((h, w), dtype=np.float32)
        resized = cv2.resize(
            cam, (x2 - x1 + 1, y2 - y1 + 1), interpolation=cv2.INTER_LINEAR
        )
        canvas = np.zeros((h, w), dtype=np.float32)
        canvas[y1 : y2 + 1, x1 : x2 + 1] = resized
        if self.blur_sigma > 0:
            canvas = gaussian_filter(canvas, sigma=self.blur_sigma)
        return self._normalize(canvas)

    def _guided_filter(self, rgb_image: np.ndarray, cam: np.ndarray) -> np.ndarray:
        if GuidedFilter is None:
            return cam
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray_t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
        cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        gf = GuidedFilter(r=16, eps=1e-3)
        refined = gf(gray_t, cam_t).squeeze().numpy()
        return self._normalize(refined)

    def _apply_heatmap(self, rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
        cam = np.clip(cam ** self.gamma, 0.0, 1.0)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(rgb, 1 - self.alpha, heatmap, self.alpha, 0)

    def _draw_detections(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: Optional[np.ndarray],
    ) -> np.ndarray:
        overlay = image.copy()
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            color = self._label_color(int(labels[idx]))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            name = self._label_name(int(labels[idx]))
            text = name
            if scores is not None:
                text = f"{name}:{float(scores[idx]):.2f}"
            cv2.putText(
                overlay,
                text,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        return overlay

    @staticmethod
    def _normalize(cam: np.ndarray) -> np.ndarray:
        min_val, max_val = cam.min(), cam.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(cam)
        return (cam - min_val) / (max_val - min_val)

    def _label_color(self, label: int) -> tuple[int, int, int]:
        rng = np.random.default_rng(abs(label) + 13)
        color = rng.integers(0, 255, size=3, dtype=np.int32)
        return int(color[0]), int(color[1]), int(color[2])

    def _label_name(self, label: int) -> str:
        if self.class_names and 1 <= label <= len(self.class_names):
            return self.class_names[label - 1]
        return f"cls{label}"


def _save_tensor_image(image: torch.Tensor, path: Path) -> None:
    array = image.detach().cpu().numpy()
    if array.ndim != 3:
        raise ValueError("期望输入CHW张量")
    array = np.clip(array, 0.0, 1.0)
    array = np.transpose(array, (1, 2, 0))
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    image_uint8 = (array * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))


def _prepare_image_batch(images: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    tensors: List[torch.Tensor] = []
    for image in images:
        if isinstance(image, torch.Tensor):
            tensors.append(image.detach().cpu().float())
        else:
            tensors.append(torch.as_tensor(image, dtype=torch.float32))
    stacked = torch.stack(tensors)
    return tensors, stacked


def run_detection_interpretability(
    model: torch.nn.Module,
    estimator,
    data_loader,
    *,
    score_threshold: float = 0.25,
    topk: int = 5,
    batch_limit: Optional[int] = None,
    gradcam_image_limit: int = 4,
    evaluation_config: Optional[Dict] = None,
    **_unused_kwargs,
) -> Optional[FidelitySummary]:
    """
    Run fidelity and optional Grad-CAM evaluation for detection models.
    """

    fidelity_cfg = {}
    if isinstance(evaluation_config, dict):
        fidelity_cfg = evaluation_config.get("model_fidelity", evaluation_config) or {}
    score_threshold = float(fidelity_cfg.get("score_threshold", score_threshold))
    topk = int(fidelity_cfg.get("topk", topk))
    iou_match_threshold = float(fidelity_cfg.get("match_iou", 0.5))
    perturbation = fidelity_cfg.get("perturbation", "blur")
    keep_mask_outside = bool(fidelity_cfg.get("keep_mask_outside", True))
    grad_cfg = fidelity_cfg.get("grad_cam") or {}
    gradcam_image_limit = int(grad_cfg.get("max_examples", gradcam_image_limit))

    evaluate_metric = os.getenv("evaluateDimension") or "interpretability"
    result_root = os.getenv("resultPath")
    if not result_root:
        raise ValueError("缺少 resultPath 环境变量，无法写入可解释性结果")

    local_output_dir = Path("..") / "evaluationData" / evaluate_metric / "output"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    remote_output_dir = Path(result_root) / evaluate_metric / "output"

    meter = DetectionFidelityMeter(
        estimator=estimator,
        score_threshold=score_threshold,
        topk=topk,
        iou_match_threshold=iou_match_threshold,
        perturbation=perturbation,
        keep_mask_outside=keep_mask_outside,
    )

    device = next(model.parameters()).device

    grad_cam_payload: List[Dict[str, str]] = []
    grad_cam_generator: Optional[GradCamGenerator] = None
        if gradcam_image_limit > 0:
            try:
                grad_cam_generator = GradCamGenerator(
                    model=model,
                    device=device,
                    output_dir=str(local_output_dir),
                )
            except Exception as exc:
                grad_cam_generator = None
                ResultSender.send_log("警告", f"Grad-CAM 初始化失败：{exc}")

    processed_images = 0
    try:
        ResultSender.send_log("进度", "开始执行检测可解释性评测")
        for batch_index, batch in enumerate(data_loader):
            if batch_limit is not None and batch_index >= batch_limit:
                break

            images, _ = batch
            if isinstance(images, torch.Tensor):
                tensor_batch = images.float()
                seq = [img.detach().cpu().float() for img in tensor_batch]
            else:
                seq, tensor_batch = _prepare_image_batch(images)

            predictions = estimator.predict(tensor_batch.to(device))
            meter.update_batch(seq, predictions)

            if grad_cam_generator and len(grad_cam_payload) < gradcam_image_limit:
                for image_tensor, pred in zip(seq, predictions):
                    if len(grad_cam_payload) >= gradcam_image_limit:
                        break
                    boxes = pred.get("boxes")
                    labels = pred.get("labels")
                    if boxes is None or labels is None or len(boxes) == 0:
                        continue
                    origin_name = f"interpretability_img_{processed_images}.png"
                    origin_local = local_output_dir / origin_name
                    _save_tensor_image(image_tensor, origin_local)
                    cam_path = grad_cam_generator.generate(
                        image_tensor,
                        boxes,
                        labels,
                        pred.get("scores"),
                        tag=f"interp_{processed_images}",
                    )
                    if cam_path is None:
                        continue
                    cam_path = Path(cam_path)
                    grad_cam_payload.append(
                        {
                            "origin": str((remote_output_dir / origin_name).as_posix()),
                            "cam": str((remote_output_dir / cam_path.name).as_posix()),
                        }
                    )
                    processed_images += 1

        summary = meter.compute()
        if summary is None:
            raise RuntimeError("No valid detections were collected; cannot compute fidelity.")

        ResultSender.send_result("fidelity_summary", summary.__dict__)
        if grad_cam_payload:
            ResultSender.send_result("grad_cam", grad_cam_payload)
        ResultSender.send_status("成功")
        ResultSender.send_log("提示", "可解释性结果已写入数据存储")
        return summary
    except Exception as exc:
        ResultSender.send_log("错误", str(exc))
        ResultSender.send_status("失败")
        raise

import pytest

torch = pytest.importorskip("torch")

from estimator import EstimatorFactory
from main_detection import main as detection_main
from model.net.dummy_detector import DummyObjectDetectionModel


class _DictOutputModel(torch.nn.Module):
    def forward(self, images):  # type: ignore[override]
        device = images[0].device if isinstance(images, list) else images.device
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device)
        scores = torch.tensor([0.95], device=device)
        labels = torch.tensor([1], dtype=torch.int64, device=device)
        return {"boxes": boxes, "scores": scores, "labels": labels}


def _make_wrapper(score_threshold: float = 0.5):
    model = DummyObjectDetectionModel(num_classes=3)
    config = {
        "framework": "pytorch",
        "task": "object_detection",
        "parameters": {"device": "cpu", "score_threshold": score_threshold},
    }
    return EstimatorFactory.create(model=model, loss=None, optimizer=None, config=config)


def test_wrapper_filters_predictions_based_on_threshold():
    estimator = _make_wrapper(score_threshold=0.6)
    batch = [torch.rand(3, 32, 32), torch.rand(3, 32, 32)]

    predictions = estimator.predict(batch)

    assert len(predictions) == 2
    # First image has low confidence and should be filtered out
    assert predictions[0]["boxes"].shape == (0, 4)
    assert predictions[1]["boxes"].shape == (1, 4)


def test_wrapper_accepts_dict_outputs():
    model = _DictOutputModel()
    config = {
        "framework": "pytorch",
        "task": "object_detection",
        "parameters": {"device": "cpu", "score_threshold": 0.0},
    }
    estimator = EstimatorFactory.create(model=model, loss=None, optimizer=None, config=config)

    batch = [torch.rand(3, 16, 16)]
    predictions = estimator.predict(batch)

    assert isinstance(predictions, list)
    assert predictions[0]["boxes"].shape == (1, 4)


def test_detection_main_runs_smoke(capsys):
    detection_main()
    captured = capsys.readouterr().out
    assert "Step 1" in captured

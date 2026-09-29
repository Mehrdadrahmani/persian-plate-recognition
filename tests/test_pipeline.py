"""Tests that need the trained models (models/) and, for end-to-end checks, the dataset (data/)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from platereader import recognizer as R
from platereader.config import REPO_ROOT

MODELS = REPO_ROOT / "models"
DATA = REPO_ROOT / "data"
needs_models = pytest.mark.skipif(not (MODELS / "lpr_config.json").exists(), reason="models/ not present")
needs_data = pytest.mark.skipif(not (DATA / "LPD" / "images").exists(), reason="data/ not present")


def test_recognizer_shapes():
    for arch in ("scratch", "pretrained"):
        m = R.build_model(arch, 22, pretrained=False).eval()
        with torch.no_grad():
            d, l = m(torch.zeros(2, 3, *R.INPUT_SIZE))
        assert d.shape == (2, 7, 10) and l.shape == (2, 22)


def test_letterbox_keeps_aspect_and_size():
    from PIL import Image

    img = Image.new("RGB", (300, 100), (255, 0, 0))
    out = R.letterbox(img)
    assert out.size == (256, 64)
    a = np.asarray(out)
    assert (a[0, 0] == 114).all() and (a[32, 128] == [255, 0, 0]).all()


def test_e2e_matching_oracle():
    """Feeding the ground truth as predictions must give perfect end-to-end metrics."""
    from platereader.e2e_eval import evaluate_at

    items = [{"image": "a.jpg", "gt_boxes": np.array([[10, 10, 60, 30], [100, 100, 160, 120.0]]), "gt_texts": ["12ب34567", "98الف76543"]}]
    cache = [{"boxes": items[0]["gt_boxes"][::-1].copy(), "det_conf": np.array([0.9, 0.8]),
              "texts": ["98الف76543", "12ب34567"], "text_conf": np.array([0.99, 0.99])}]
    m, plates, _ = evaluate_at(items, cache, 0.5)
    assert m["e2e_precision"] == m["e2e_recall"] == 1.0 and m["cer_on_detected"] == 0.0 and plates.correct.all()


@needs_models
def test_trained_recognizer_loads_and_matches_torchscript():
    from platereader.evaluate import load_recognizer

    model, ck = load_recognizer(MODELS / "lpr_scratch_best.pt")
    ts = torch.jit.load(str(MODELS / "lpr_best.ts"))
    x = torch.randn(2, 3, *R.INPUT_SIZE)
    with torch.no_grad():
        a, b = model(x), ts(x)
    assert max((p - q).abs().max().item() for p, q in zip(a, b)) < 1e-4
    assert len(ck["letter_vocab"]) == 22


@needs_models
@needs_data
@pytest.mark.parametrize("backend", ["torch", "onnx"])
def test_pipeline_reads_known_plate(backend):
    from platereader.pipeline import PlateRecognizer

    reader = PlateRecognizer(MODELS, backend=backend)
    res = reader.predict(DATA / "LPD" / "images" / "000002.jpg")
    assert res and res[0].text == "78س92616"
    assert Path(__file__).exists()

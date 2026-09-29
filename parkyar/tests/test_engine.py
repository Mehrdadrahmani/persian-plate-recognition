from pathlib import Path

import cv2
import numpy as np
import pytest

from parkyar.engine import PlateEngine

SAMPLE = Path(__file__).parent / "sample_car.jpg"


def test_engine_runs_on_blank_image():
    eng = PlateEngine()
    assert eng.read(np.full((480, 640, 3), 128, np.uint8)) == []


@pytest.mark.skipif(not SAMPLE.exists(), reason="no sample image")
def test_engine_reads_sample():
    eng = PlateEngine()
    res = eng.read(cv2.imread(str(SAMPLE)))
    assert res and res[0].text == "94س76947"

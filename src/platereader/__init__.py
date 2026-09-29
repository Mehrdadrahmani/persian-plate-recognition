"""platereader: Iranian licence plate detection (YOLO26) and recognition (CNN) with an end-to-end pipeline."""

__version__ = "1.0.0"
__all__ = ["normalize_plate", "parse_plate", "to_latin", "to_persian_display", "PlateRecognizer"]

from .plates import normalize_plate, parse_plate, to_latin, to_persian_display  # noqa: F401,E402


def __getattr__(name):  # lazy import so `import platereader` does not load torch/ultralytics
    if name == "PlateRecognizer":
        from .pipeline import PlateRecognizer

        return PlateRecognizer
    raise AttributeError(name)

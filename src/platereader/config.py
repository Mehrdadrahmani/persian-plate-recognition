"""Configuration: YAML settings + resolved paths + device selection."""
import copy
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v


@dataclass
class Paths:
    root: Path
    data_dir: Path
    work_dir: Path
    models_dir: Path
    reports_dir: Path
    tag: str = "full"

    @property
    def lpd_dir(self):
        return self.data_dir / "LPD"

    @property
    def lpr_dir(self):
        return self.data_dir / "LPR"

    @property
    def splits_dir(self):
        return self.work_dir / "splits"

    @property
    def lpd_yolo_dir(self):
        return self.work_dir / "lpd_yolo"

    @property
    def runs_dir(self):
        return self.work_dir / "runs"

    def reports(self, phase):
        """reports/<phase>[_smoke]/ with a figures/ subfolder."""
        d = self.reports_dir / (phase if self.tag == "full" else f"{phase}_{self.tag}")
        (d / "figures").mkdir(parents=True, exist_ok=True)
        return d


@dataclass
class Config:
    raw: dict
    paths: Paths
    smoke: bool = False
    extra: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.raw[key]

    @property
    def seed(self):
        return self.raw["seed"]


def load_config(path=None, smoke=False, data_dir=None, models_dir=None, work_dir=None):
    """Load configs/default.yaml (or `path`); apply smoke overrides; resolve paths against the repo root."""
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    overrides = raw.pop("smoke_overrides", {})
    if smoke:
        _deep_update(raw, overrides)
    raw["lpd"].setdefault("smoke_n", None)
    raw["lpr"].setdefault("smoke_n", None)
    if not smoke:
        raw["lpd"]["smoke_n"] = raw["lpr"]["smoke_n"] = None

    def resolve(p):
        p = Path(os.path.expanduser(str(p)))
        return p if p.is_absolute() else (REPO_ROOT / p)

    pc = raw["paths"]
    paths = Paths(
        root=REPO_ROOT,
        data_dir=resolve(data_dir or os.environ.get("PLATE_DATA_DIR", pc["data_dir"])),
        work_dir=resolve(work_dir or pc["work_dir"]),
        models_dir=resolve(models_dir or os.environ.get("PLATE_MODELS_DIR", pc["models_dir"])),
        reports_dir=resolve(pc["reports_dir"]),
        tag="smoke" if smoke else "full",
    )
    for d in (paths.work_dir, paths.reports_dir, paths.splits_dir):
        d.mkdir(parents=True, exist_ok=True)
    return Config(raw=raw, paths=paths, smoke=smoke)


def add_common_args(parser):
    """--config / --smoke / --data-dir / --models-dir / --device shared by all scripts."""
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--smoke", action="store_true", help="tiny subset + 1 epoch, to check that everything runs")
    parser.add_argument("--data-dir", default=None, help="folder containing LPD/ and LPR/ (default: data/)")
    parser.add_argument("--models-dir", default=None, help="trained model bundle (default: models/)")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    return parser


def config_from_args(args):
    return load_config(args.config, smoke=args.smoke, data_dir=args.data_dir, models_dir=args.models_dir)

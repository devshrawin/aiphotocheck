from __future__ import annotations

from pathlib import Path

from .base import Detector, DetectorResult  # noqa: F401
from .classical import ClassicalDetector
from .hf_classifier import HFClassifierDetector
from .seed import SeedDetector
from .trufor import TruForDetector


def build_detectors(cfg: dict, project_root: Path, names: list[str] | None = None) -> list[Detector]:
    names = names or cfg["detectors"]
    out: list[Detector] = []
    for n in names:
        if n == "classical":
            out.append(ClassicalDetector())
        elif n in cfg.get("hf_models", {}):
            m = cfg["hf_models"][n]
            out.append(HFClassifierDetector(n, m["model_id"], m["fake_label_hints"], m.get("onnx")))
        elif n == "trufor":
            out.append(TruForDetector(cfg["trufor"], project_root))
        elif n == "seed":
            out.append(SeedDetector(cfg["seed"], project_root))
        else:
            raise ValueError(f"unknown detector {n!r}")
    return out

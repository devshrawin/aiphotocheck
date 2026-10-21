from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DetectorResult:
    detector: str
    score: float | None = None       # 0..1, higher = more likely manipulated/generated. NOT a calibrated probability.
    heatmap_path: str | None = None
    note: str = ""
    extra: dict = field(default_factory=dict)


class Detector:
    name = "base"

    def available(self) -> tuple[bool, str]:
        """(is_usable, reason_if_not)."""
        return True, ""

    def score(self, image_path: Path, outdir: Path) -> DetectorResult:  # pragma: no cover
        raise NotImplementedError

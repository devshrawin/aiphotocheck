"""TruFor adapter (external repo, run as a subprocess).

Runs TruFor's inference-only script test_docker/src/trufor_test.py on CPU. The adapter expects TruFor to write
<outdir>/<image filename>.npz containing 'score' (image-level integrity score) and 'map'
(pixel-level localisation map).

LICENCE: TruFor permits informational / non-profit use only. Get permission before production use.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .base import Detector, DetectorResult


class TruForDetector(Detector):
    name = "trufor"

    def __init__(self, cfg: dict, project_root: Path):
        self.repo = (project_root / cfg["repo_dir"]).resolve()
        self.cmd = cfg["command"]
        self.cwd = (self.repo / cfg["cwd"]) if cfg.get("cwd") else self.repo
        self.weights = (self.repo / cfg["weights"]) if cfg.get("weights") else None

    def available(self):
        if not self.repo.exists():
            return False, f"TruFor repo not found at {self.repo}; run setup.sh"
        if not (self.weights.exists() if self.weights else list(self.repo.rglob("*.pth*"))):
            return False, "TruFor weights not found; follow the download steps in the TruFor README"
        return True, ""

    def score(self, image_path: Path, outdir: Path) -> DetectorResult:
        image_path = Path(image_path).resolve()
        outdir = Path(outdir).resolve() / "trufor"
        outdir.mkdir(parents=True, exist_ok=True)
        cmd = self.cmd.format(repo=shlex.quote(str(self.repo)), input=shlex.quote(str(image_path)), outdir=shlex.quote(str(outdir)))
        cmd = cmd.replace("python ", f"{shlex.quote(sys.executable)} ", 1)
        # The official checkpoint is a pickled file with numpy scalars; torch>=2.6 defaults to weights_only=True and
        # rejects it. Allow full unpickling for this trusted, md5-checked file only (child process only).
        env = {**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.cwd, env=env)
        npz = outdir / (Path(image_path).name + ".npz")
        if r.returncode != 0 or not npz.exists():
            return DetectorResult(self.name, None, note=f"TruFor run failed: {(r.stderr or r.stdout)[-300:]}")
        d = np.load(npz, allow_pickle=False)
        hm = d["map"].astype(np.float32)
        hp = None
        if hm.size:
            from matplotlib import cm
            col = (cm.inferno(np.clip(hm, 0, 1))[:, :, :3] * 255).astype(np.uint8)
            hp = outdir / f"{Path(image_path).stem}_trufor_heatmap.png"
            Image.fromarray(col).save(hp)
        if "score" not in d.files:
            return DetectorResult(self.name, None, note="TruFor output has no image-level 'score'")
        return DetectorResult(self.name, float(d["score"]), str(hp) if hp else None)

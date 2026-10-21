"""Classical forensics baseline: Error Level Analysis + noise-residual inconsistency.

Cheap, explainable and always available, but weak against modern AI edits. Its job is (a) to give
the officer a heat map and (b) to be the baseline every ML detector must beat.
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .base import Detector, DetectorResult


def ela_map(img: Image.Image, quality: int = 90) -> np.ndarray:
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    re_ = Image.open(buf).convert("RGB")
    d = np.abs(np.asarray(rgb, np.int16) - np.asarray(re_, np.int16)).max(axis=2).astype(np.float32)
    return cv2.GaussianBlur(d, (0, 0), 1.5)


def residual_map(gray: np.ndarray) -> np.ndarray:
    g8 = gray.astype(np.uint8)
    return np.abs(g8.astype(np.float32) - cv2.medianBlur(g8, 3).astype(np.float32))


def _blocks(m: np.ndarray, gray: np.ndarray, block: int, active_thresh: float = 8.0):
    h, w = m.shape
    H, W = h // block * block, w // block * block
    mb = m[:H, :W].reshape(H // block, block, W // block, block).mean(axis=(1, 3))
    gb = gray[:H, :W].reshape(H // block, block, W // block, block).std(axis=(1, 3))
    return mb, gb > active_thresh


def robust_z(mb: np.ndarray, active: np.ndarray, z_thr: float = 4.0):
    vals = mb[active]
    if vals.size < 20:
        return 0.0, 0.0, np.zeros_like(mb)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-6
    z = (mb - med) / mad
    z[~active] = 0
    return float((z > z_thr).sum()) / vals.size, float(z.max()), z


class ClassicalDetector(Detector):
    name = "classical"

    def __init__(self, block: int = 32):
        self.block = block

    def score(self, image_path: Path, outdir: Path) -> DetectorResult:
        img = Image.open(image_path)
        if max(img.size) > 4000:
            s = 4000 / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
        rgb = img.convert("RGB")
        gray = np.asarray(rgb.convert("L"), np.float32)

        zs, fr, zm = [], [], []
        for m in (ela_map(rgb), residual_map(gray)):
            mb, active = _blocks(m, gray, self.block)
            frac, zmax, z = robust_z(mb, active)
            zs.append(z); fr.append(frac); zm.append(zmax)

        sub = [0.5 * min(1.0, f * 15) + 0.5 * min(1.0, max(0.0, zmx - 3) / 12) for f, zmx in zip(fr, zm)]
        score = float(max(sub))

        zc = np.maximum(zs[0], zs[1])
        hm = cv2.resize(np.clip(zc / 12.0, 0, 1), (rgb.width, rgb.height), interpolation=cv2.INTER_NEAREST)
        color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]
        overlay = (0.55 * np.asarray(rgb) + 0.45 * color).astype(np.uint8)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        hp = outdir / f"{Path(image_path).stem}_classical_heatmap.png"
        Image.fromarray(overlay).save(hp)
        return DetectorResult(self.name, score, str(hp),
                              extra={"ela_outlier_frac": fr[0], "ela_zmax": zm[0], "noise_outlier_frac": fr[1], "noise_zmax": zm[1]})

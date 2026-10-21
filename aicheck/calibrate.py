"""Measure the detectors on labelled real/AI files and fit a calibration so the percentage means something."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from . import files
from .pipeline import detector_scores

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".pdf", ".docx", ".pptx", ".xlsx"}


def list_files(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in SUPPORTED and not p.name.startswith("."))


def score_folder(folder: Path, detectors, work: Path, limit: int | None = None) -> pd.DataFrame:
    rows = []
    paths = list_files(folder)[:limit]
    for i, f in enumerate(paths, 1):
        _, scores, _ = detector_scores(files.extract_images(f, files.sniff(f), work), detectors, work)
        rows.append({"path": str(f), **scores})
        if i % 25 == 0:
            print(f"  {i}/{len(paths)} files in {folder}", flush=True)
    return pd.DataFrame(rows)


def _tpr_at(fpr, tpr, target):
    ok = np.where(fpr <= target)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def _metrics(y, s) -> dict:
    fpr, tpr, _ = roc_curve(y, s)
    return {"auc": float(roc_auc_score(y, s)), "tpr@fpr5%": _tpr_at(fpr, tpr, 0.05), "tpr@fpr1%": _tpr_at(fpr, tpr, 0.01)}


def calibrate(real_dir: Path, ai_dir: Path, detectors, out_dir: Path, limit: int | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    real, ai = score_folder(real_dir, detectors, out_dir / "work", limit), score_folder(ai_dir, detectors, out_dir / "work", limit)
    if real.empty or ai.empty:
        raise SystemExit(f"need supported files in both folders (real={len(real)}, ai={len(ai)})")
    df = pd.concat([real.assign(label=0), ai.assign(label=1)], ignore_index=True)
    df.to_csv(out_dir / "scores.csv", index=False)

    names = [d.name for d in detectors if d.name in df.columns]
    metrics = {}
    for n in names:
        m = df[n].notna()
        if df.label[m].nunique() == 2:
            metrics[n] = {**_metrics(df.label[m], df.loc[m, n]), "n": int(m.sum())}
    ok = df[names].notna().all(axis=1)
    X, y = df.loc[ok, names].to_numpy(), df.label[ok].to_numpy()
    clf = LogisticRegression(C=1.0, class_weight="balanced").fit(X, y)
    calib = {"detectors": names, "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
             "n_real": int((y == 0).sum()), "n_ai": int((y == 1).sum())}
    p = clf.predict_proba(X)[:, 1]
    metrics["combined"] = {**_metrics(y, p), "n": int(ok.sum()), "note": "measured on the same files used to fit; optimistic"}
    calib["metrics"] = metrics
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2))
    return calib

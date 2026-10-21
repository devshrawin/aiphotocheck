"""AI-generated-image likelihood: score an image, calibrate the scores on labelled real/AI folders.

Only the image-level AI detectors are used (no tamper-localisation), so a scan takes about a second.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

AI_DETECTORS = ["community_forensics", "sdxl_detector"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def score_files(files: list[Path], detectors, work: Path) -> pd.DataFrame:
    rows = []
    for f in files:
        row = {"path": str(f)}
        for d in detectors:
            r = d.score(f, work)
            row[d.name] = r.score
        rows.append(row)
    return pd.DataFrame(rows)


def _tpr_at(fpr, tpr, target):
    ok = np.where(fpr <= target)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def evaluate(real_dir: Path, ai_dir: Path, detectors, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    real, ai = _images(real_dir), _images(ai_dir)
    if not real or not ai:
        raise SystemExit(f"need images in both folders (real={len(real)}, ai={len(ai)})")
    df = pd.concat([
        score_files(real, detectors, out_dir / "work").assign(label=0),
        score_files(ai, detectors, out_dir / "work").assign(label=1),
    ], ignore_index=True)
    df.to_csv(out_dir / "scores.csv", index=False)

    names = [d.name for d in detectors]
    metrics = {}
    for n in names:
        m = df[n].notna()
        fpr, tpr, _ = roc_curve(df.label[m], df.loc[m, n])
        metrics[n] = {"auc": float(roc_auc_score(df.label[m], df.loc[m, n])),
                      "tpr@fpr5%": _tpr_at(fpr, tpr, 0.05), "tpr@fpr1%": _tpr_at(fpr, tpr, 0.01), "n": int(m.sum())}

    # Logistic regression on the detector scores turns them into a probability that matches observed frequencies.
    ok = df[names].notna().all(axis=1)
    X, y = df.loc[ok, names].to_numpy(), df.label[ok].to_numpy()
    clf = LogisticRegression(C=1.0, class_weight="balanced").fit(X, y)
    calib = {"detectors": names, "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
             "n_real": int((y == 0).sum()), "n_ai": int((y == 1).sum())}
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2))
    p = clf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, p)
    metrics["combined"] = {"auc": float(roc_auc_score(y, p)), "tpr@fpr5%": _tpr_at(fpr, tpr, 0.05),
                           "tpr@fpr1%": _tpr_at(fpr, tpr, 0.01), "n": int(ok.sum())}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def likelihood(image: Path, detectors, calib_path: Path | None, work: Path) -> dict:
    scores = {d.name: d.score(Path(image), work).score for d in detectors}
    have = {k: v for k, v in scores.items() if v is not None}
    if calib_path and Path(calib_path).exists():
        c = json.loads(Path(calib_path).read_text())
        if all(n in have for n in c["detectors"]):
            z = c["intercept"] + sum(w * have[n] for w, n in zip(c["coef"], c["detectors"]))
            return {"ai_likelihood_pct": round(100 / (1 + np.exp(-z)), 1), "calibrated": True, "detector_scores": scores}
    pct = round(100 * float(np.mean(list(have.values()))), 1) if have else None
    return {"ai_likelihood_pct": pct, "calibrated": False, "detector_scores": scores,
            "warning": "Uncalibrated: this is an average of raw detector scores, not a probability."}

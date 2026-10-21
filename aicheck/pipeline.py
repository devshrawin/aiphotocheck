"""Analyse one attachment (image, PDF, Office file) and report how likely it is to be AI-generated or AI-edited.

The result is decision support for a human reviewer. A band is a routing hint, never an automatic rejection.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import files, provenance

SUGGESTED_ACTION = {
    "low": "No AI signals found. This does not prove the file is genuine.",
    "review": "Some signals. Ask for the original file or verify with the source before relying on it.",
    "high": "Strong signals. Verify directly with the source. Do not reject on this flag alone.",
}


def noisy_or(weights) -> float:
    p = 1.0
    for w in weights:
        p *= 1.0 - max(0.0, min(1.0, w))
    return 1.0 - p


def load_calibration(path: Path | None) -> dict | None:
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else None


def detector_scores(images, detectors, workdir: Path) -> tuple[dict, dict, list[str]]:
    """Per detector: the highest score over all images in the file. Returns (details, scores, heatmaps)."""
    details: dict[str, dict] = {}
    scores: dict[str, float] = {}
    heatmaps: list[str] = []
    for d in detectors:
        ok, why = d.available()
        if not ok:
            details[d.name] = {"available": False, "note": why}
            continue
        results = []
        for p, _ in images:
            try:
                results.append(d.score(p, workdir))
            except Exception as e:  # one bad image must not sink the report
                details[d.name] = {"available": True, "error": str(e)[:200]}
        scored = [r for r in results if r.score is not None]
        if scored:
            best = max(scored, key=lambda r: r.score)
            details[d.name] = {"available": True, "score": round(best.score, 4), "note": best.note}
            scores[d.name] = best.score
            heatmaps += [r.heatmap_path for r in scored if r.heatmap_path]
    return details, scores, heatmaps


def ml_component(scores: dict, cfg: dict, calib: dict | None) -> tuple[float, bool]:
    """(probability-like value from the detectors, whether it is calibrated)."""
    if calib and all(n in scores for n in calib["detectors"]):
        z = calib["intercept"] + sum(w * scores[n] for w, n in zip(calib["coef"], calib["detectors"]))
        return 1 / (1 + math.exp(-z)), True
    # Uncalibrated: a detector only counts when it leans "fake" (> 0.5). Scores below that are the model saying "real",
    # and some models sit at a constant sub-0.5 value for everything, which must not become a floor on the percentage.
    wts = cfg["detector_weights"]
    leaning = [max(0.0, (s - 0.5) * 2) * wts.get(n, 1.0) for n, s in scores.items()]
    return (max(leaning) if leaning else 0.0) * cfg["uncalibrated_ml_discount"], False


def analyze(path, cfg: dict, detectors, workdir, calib: dict | None = None) -> dict:
    path, workdir = Path(path), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    kind = files.sniff(path)
    findings = provenance.file_checks(path, kind)
    images = files.extract_images(path, kind, workdir)
    if kind == "pdf" or kind in files.OFFICE_MEDIA_DIRS:  # generator metadata often survives inside embedded images
        for p, origin in images:
            if origin == "embedded":
                findings += [f for f in provenance.file_checks(p, files.sniff(p)) if f.kind == "ai"]

    details, scores, heatmaps = detector_scores(images, detectors, workdir)
    ml, calibrated = ml_component(scores, cfg, calib)
    ai_evidence = [f for f in findings if f.kind == "ai" and f.weight > 0]
    edit_evidence = [f for f in findings if f.kind == "edit" and f.weight > 0]
    likelihood = noisy_or([f.weight for f in ai_evidence] + [ml])
    th = cfg["thresholds"]
    band = "high" if likelihood >= th["high"] else "review" if likelihood >= th["review"] else "low"
    edit_score = noisy_or([f.weight for f in edit_evidence])
    if band == "low" and edit_score >= 0.4:
        band = "review"

    reasons = [f.message for f in sorted(ai_evidence, key=lambda f: -f.weight)]
    if scores and max(scores.values()) > 0.5:
        top = max(scores, key=scores.get)
        reasons.append(f"Detector '{top}' scored {scores[top]:.2f}" + ("" if calibrated else " (detectors are uncalibrated)") + ".")
    return {
        "file": str(path), "file_type": kind, "images_analyzed": len(images),
        "ai_likelihood": round(likelihood, 4), "ai_likelihood_pct": round(100 * likelihood, 1), "calibrated": calibrated,
        "band": band, "suggested_action": SUGGESTED_ACTION[band],
        "reasons": reasons,
        "other_alteration_signals": [f.message for f in sorted(edit_evidence, key=lambda f: -f.weight)],
        "context": [f.message for f in findings if f.kind == "info"],
        "detectors": details, "heatmaps": heatmaps,
        "caveat": None if calibrated else "Percentage is NOT calibrated: it is a heuristic score, not a measured probability. "
                                           "Run `aicheck calibrate` on labelled real/AI files first.",
    }


def save_report(report: dict, outdir: Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / (Path(report["file"]).stem + ".json")
    p.write_text(json.dumps(report, indent=2, default=str))
    return p

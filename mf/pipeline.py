"""Run the rules layer + content checks + detectors on one upload and produce a risk report.

Design rules (from the product discussion):
  * The CITIZEN never sees a score or the words 'AI' / 'altered'. They get `citizen_message`.
  * The OFFICER sees `officer_view`: band, reasons, per-detector scores, heat maps.
  * A band is a routing decision for human review. It is never an automatic rejection.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import marksheet, rules

CITIZEN_MESSAGES = {
    "low": "Your document has been received.",
    "review": "We could not automatically verify this document. Please upload a clearer copy, or fetch it directly "
              "from the issuing authority (for example through DigiLocker). Your application is not rejected.",
    "high": "Your document needs a routine manual verification. A verification officer may contact you. "
            "Your application has not been rejected.",
}


def noisy_or(weights: list[float]) -> float:
    p = 1.0
    for w in weights:
        p *= 1.0 - max(0.0, min(1.0, w))
    return 1.0 - p


def combine(findings, det_scores: dict[str, float], cfg: dict) -> dict:
    rules_score = noisy_or([f.weight for f in findings])
    wts = cfg["detector_weights"]
    weighted = {n: s * wts.get(n, 1.0) for n, s in det_scores.items() if s is not None}
    det_score = max(weighted.values()) if weighted else 0.0
    combined = noisy_or([rules_score, det_score])
    th = cfg["thresholds"]
    band = "high" if combined >= th["high"] else "review" if combined >= th["review"] else "low"
    return {"rules_score": rules_score, "detector_score": det_score, "combined_score": combined, "band": band}


def analyze(path, cfg: dict, detectors, workdir, expected: dict | None = None) -> dict:
    path, workdir = Path(path), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    kind = rules.sniff(path)
    findings = rules.file_checks(path, kind)
    images = rules.extract_images(path, kind, workdir)
    if kind == "pdf":
        for p, origin in images:
            if origin == "embedded":
                findings += [f for f in rules.image_checks(p) if f.id != "no_exif"]

    ms_findings, ms_info = marksheet.run_checks(path, kind, images, cfg, expected)
    findings += ms_findings

    det_out: dict[str, dict] = {}
    det_scores: dict[str, float] = {}
    heatmaps: list[str] = []
    for d in detectors:
        ok, why = d.available()
        if not ok:
            det_out[d.name] = {"available": False, "note": why}
            continue
        per = []
        for p, _ in images:
            try:
                per.append(d.score(p, workdir))
            except Exception as e:  # one bad image must not sink the report
                det_out[d.name] = {"available": True, "error": str(e)[:200]}
        scored = [r for r in per if r.score is not None]
        if scored:
            best = max(scored, key=lambda r: r.score)
            det_out[d.name] = {"available": True, "score": best.score, "extra": best.extra, "note": best.note}
            det_scores[d.name] = best.score
            heatmaps += [r.heatmap_path for r in scored if r.heatmap_path]

    c = combine(findings, det_scores, cfg)
    reasons = [f.message for f in sorted(findings, key=lambda f: -f.weight) if f.weight > 0]
    return {
        "file": str(path), "file_type": kind, "images_analyzed": len(images),
        "band": c["band"],
        "scores": {k: round(v, 4) for k, v in c.items() if k != "band"},
        "detectors": det_out,
        "findings": [f.to_dict() for f in findings],
        "ocr": {"source": ms_info.get("source"), "parsed": ms_info.get("parsed")},
        "citizen_message": CITIZEN_MESSAGES[c["band"]],
        "officer_view": {
            "band": c["band"], "reasons": reasons, "heatmaps": heatmaps,
            "suggested_action": {"low": "No action needed.",
                                 "review": "Compare with the original; request a clearer copy or verify with the issuer.",
                                 "high": "Verify directly with the issuing authority before any decision. Do not reject on this flag alone."}[c["band"]],
        },
    }


def save_report(report: dict, outdir: Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / (Path(report["file"]).stem + ".json")
    p.write_text(json.dumps(report, indent=2, default=str))
    return p

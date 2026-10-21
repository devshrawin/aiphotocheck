from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATION = ROOT / "calibration.json"


def load_cfg(path: str | None) -> dict:
    return json.loads(Path(path or ROOT / "config.json").read_text())


def cmd_doctor(args):
    from .detectors import build_detectors

    cfg = load_cfg(args.config)
    print("System tools")
    print(f"  pdftoppm   {'ok' if shutil.which('pdftoppm') else 'MISSING (brew install poppler); needed for text-only PDF pages'}")
    print("Detectors")
    for d in build_detectors(cfg, ROOT, deep=True):
        ok, why = d.available()
        print(f"  {d.name:22s} {'ready' if ok else 'not ready: ' + why}")
    print(f"Calibration  {'found: ' + str(DEFAULT_CALIBRATION) if DEFAULT_CALIBRATION.exists() else 'none yet (percentages will be marked uncalibrated)'}")


def _expand(paths: list[str]) -> list[Path]:
    from .calibrate import list_files

    out: list[Path] = []
    for p in map(Path, paths):
        if not p.exists():
            sys.exit(f"aicheck: no such file or folder: {p}")
        out += list_files(p) if p.is_dir() else [p]
    return out


def cmd_scan(args):
    from .detectors import build_detectors
    from .pipeline import analyze, load_calibration, save_report

    cfg = load_cfg(args.config)
    dets = build_detectors(cfg, ROOT, args.detectors.split(",") if args.detectors else None, deep=args.deep)
    calib = load_calibration(Path(args.calibration) if args.calibration else DEFAULT_CALIBRATION)
    out = Path(args.out)
    reports = []
    for f in _expand(args.files):
        rep = analyze(f, cfg, dets, out / "work", calib)
        save_report(rep, out / "reports")
        reports.append(rep)
        if not args.json:
            tag = "" if rep["calibrated"] else " (uncalibrated)"
            print(f"\n{f}\n  AI likelihood: {rep['ai_likelihood_pct']}%{tag}   band: {rep['band'].upper()}   type: {rep['file_type']}")
            for r in rep["reasons"]:
                print(f"   - {r}")
            for r in rep["other_alteration_signals"]:
                print(f"   ~ other alteration signal: {r}")
            for h in rep["heatmaps"][:3]:
                print(f"   heatmap: {h}")
    if args.json:
        print(json.dumps(reports, indent=2, default=str))
    elif reports:
        print(f"\nReports saved in {out / 'reports'}")


def cmd_calibrate(args):
    from .calibrate import calibrate
    from .detectors import build_detectors

    cfg = load_cfg(args.config)
    dets = build_detectors(cfg, ROOT, args.detectors.split(",") if args.detectors else None, deep=args.deep)
    calib = calibrate(Path(args.real), Path(args.ai), dets, Path(args.out), args.limit)
    for name, m in calib["metrics"].items():
        print(f"{name:22s} auc={m['auc']:.3f}  tpr@fpr5%={m['tpr@fpr5%']:.3f}  tpr@fpr1%={m['tpr@fpr1%']:.3f}  n={m['n']}")
    src = Path(args.out) / "calibration.json"
    print(f"\nCalibration written to {src}")
    print(f"To use it for scans: cp {src} {DEFAULT_CALIBRATION}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aicheck", description="Check whether a document, picture or attachment was generated or altered with AI")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="show which detectors are ready").set_defaults(fn=cmd_doctor)

    p = sub.add_parser("scan", help="check one or more files, or whole folders")
    p.add_argument("files", nargs="+")
    p.add_argument("--deep", action="store_true", help="also run the slow detectors (TruFor, SEED): minutes per file")
    p.add_argument("--detectors", default=None, help="comma-separated override")
    p.add_argument("--calibration", default=None); p.add_argument("--out", default="results")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON only")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("calibrate", help="measure detectors on labelled folders and fit a calibration")
    p.add_argument("--real", required=True); p.add_argument("--ai", required=True)
    p.add_argument("--deep", action="store_true"); p.add_argument("--detectors", default=None)
    p.add_argument("--out", default="results_calibration"); p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_calibrate)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

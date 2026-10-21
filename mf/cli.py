from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_cfg(path: str | None) -> dict:
    return json.loads(Path(path or ROOT / "config.json").read_text())


def cmd_doctor(args):
    from .detectors import build_detectors

    cfg = load_cfg(args.config)
    print("System tools")
    for t in ("tesseract", "pdftoppm"):
        print(f"  {t:10s} {'ok' if shutil.which(t) else 'MISSING (brew install tesseract poppler)'}")
    print("Python packages")
    for m in ("numpy", "PIL", "cv2", "pypdf", "pikepdf", "sklearn", "pandas", "matplotlib", "reportlab", "pytesseract", "torch", "transformers"):
        try:
            __import__(m)
            print(f"  {m:12s} ok")
        except Exception:
            print(f"  {m:12s} MISSING" + ("  (needed only for ML detectors)" if m in ("torch", "transformers") else ""))
    print("Detectors")
    for d in build_detectors(cfg, ROOT):
        ok, why = d.available()
        print(f"  {d.name:22s} {'ready' if ok else 'not ready: ' + why}")


def cmd_make_samples(args):
    from .synth import make_dataset

    p = make_dataset(Path(args.out), n=args.n, seed=args.seed)
    print(f"Wrote {p} (+ files, registry.json)")


def cmd_scan(args):
    from .detectors import build_detectors
    from .pipeline import analyze, save_report

    cfg = load_cfg(args.config)
    dets = build_detectors(cfg, ROOT, args.detectors.split(",") if args.detectors else None)
    expected = json.loads(args.expect) if args.expect else None
    rep = analyze(args.file, cfg, dets, Path(args.out) / "work", expected)
    p = save_report(rep, Path(args.out) / "reports")
    print(f"Band: {rep['band']}   scores: {rep['scores']}")
    print("Officer view:")
    for r in rep["officer_view"]["reasons"] or ["(no findings)"]:
        print("  -", r)
    print("Citizen message:", rep["citizen_message"])
    print("Full report:", p)


def cmd_eval(args):
    from .detectors import build_detectors
    from .evaluate import run, write_outputs

    cfg = load_cfg(args.config)
    dets = build_detectors(cfg, ROOT, args.detectors.split(",") if args.detectors else None)
    df = run(Path(args.data), Path(args.out), cfg, dets, args.limit)
    met = write_outputs(df, Path(args.out), cfg)
    allm = met[met.scope == "all"][["score", "auc", "tpr@fpr5%", "tpr@fpr1%"]]
    print(allm.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nReport: {Path(args.out) / 'report.md'}")


def _ai_detectors(args):
    from .aiprob import AI_DETECTORS
    from .detectors import build_detectors

    return build_detectors(load_cfg(args.config), ROOT, AI_DETECTORS)


def cmd_ai_eval(args):
    from .aiprob import evaluate

    met = evaluate(Path(args.real), Path(args.ai), _ai_detectors(args), Path(args.out))
    for name, m in met.items():
        print(f"{name:22s} auc={m['auc']:.3f}  tpr@fpr5%={m['tpr@fpr5%']:.3f}  tpr@fpr1%={m['tpr@fpr1%']:.3f}  n={m['n']}")
    print(f"\nCalibration: {Path(args.out) / 'calibration.json'}")


def cmd_ai_score(args):
    from .aiprob import likelihood

    res = likelihood(Path(args.file), _ai_detectors(args), Path(args.calibration) if args.calibration else None, Path(args.out) / "work")
    print(json.dumps(res, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mf", description="Marksheet forensics harness")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check tools, packages and detector availability").set_defaults(fn=cmd_doctor)
    p = sub.add_parser("make-samples", help="generate a synthetic labelled dataset")
    p.add_argument("--out", default="data"); p.add_argument("--n", type=int, default=12); p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_make_samples)
    p = sub.add_parser("scan", help="analyse one file")
    p.add_argument("file"); p.add_argument("--out", default="results"); p.add_argument("--detectors", default=None)
    p.add_argument("--expect", default=None, help='registry record as JSON, e.g. \'{"name":"A B","roll":"123456","dob":"01-02-2007"}\'')
    p.set_defaults(fn=cmd_scan)
    p = sub.add_parser("ai-eval", help="AI-generated vs real images: evaluate detectors and fit a calibration")
    p.add_argument("--real", required=True); p.add_argument("--ai", required=True); p.add_argument("--out", default="results_ai")
    p.set_defaults(fn=cmd_ai_eval)
    p = sub.add_parser("ai-score", help="AI-generated likelihood (%) for one image")
    p.add_argument("file"); p.add_argument("--calibration", default=None); p.add_argument("--out", default="results_ai")
    p.set_defaults(fn=cmd_ai_score)
    p = sub.add_parser("eval", help="evaluate on a labelled folder")
    p.add_argument("--data", default="data"); p.add_argument("--out", default="results"); p.add_argument("--detectors", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_eval)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

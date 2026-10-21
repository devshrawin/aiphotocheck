"""Run the pipeline over a labelled folder and compare detectors/rules/combined on the same files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    def tqdm(it, **_kw):
        return it

from .pipeline import analyze, save_report

# Okabe-Ito: colour-blind-safe, assigned in fixed order per column (never cycled).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def run(data_dir: Path, out_dir: Path, cfg: dict, detectors, limit: int | None = None) -> pd.DataFrame:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    labels = pd.read_csv(data_dir / "labels.csv")
    if limit:
        labels = labels.sample(n=min(limit, len(labels)), random_state=0).reset_index(drop=True)
    registry = json.loads((data_dir / "registry.json").read_text()) if (data_dir / "registry.json").exists() else {}
    rows, unavailable = [], {}
    for r in tqdm(labels.itertuples(index=False), total=len(labels), desc="analyzing"):
        rep = analyze(data_dir / r.path, cfg, detectors, out_dir / "work", registry.get(str(r.sample_id)))
        save_report(rep, out_dir / "reports")
        row = {"path": r.path, "label": r.label, "edit_type": r.edit_type, "carrier": r.carrier,
               "rules_score": rep["scores"]["rules_score"], "combined_score": rep["scores"]["combined_score"], "band": rep["band"]}
        for name, d in rep["detectors"].items():
            if d.get("available") is False:
                unavailable[name] = d.get("note", "")
            elif "score" in d:
                row[f"det_{name}"] = d["score"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["unavailable"] = unavailable
    return df


def _tpr_at(fpr, tpr, target):
    ok = np.where(fpr <= target)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def _metrics(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if len(set(y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y, s)
    return {"auc": float(roc_auc_score(y, s)), "tpr@fpr5%": _tpr_at(fpr, tpr, 0.05), "tpr@fpr1%": _tpr_at(fpr, tpr, 0.01)}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    y = (df["label"] == "forged").astype(int)
    cols = ["rules_score"] + [c for c in df.columns if c.startswith("det_")] + ["combined_score"]
    out = []
    for c in cols:
        m = df[c].notna()
        res = _metrics(y[m], df.loc[m, c])
        if res:
            out.append({"score": c, "scope": "all", "n_forged": int(y[m].sum()), "n_genuine": int((1 - y[m]).sum()), **res})
        for et in sorted(set(df.loc[df.label == "forged", "edit_type"])):
            mm = m & ((df.edit_type == et) | (df.label == "genuine"))
            res = _metrics(y[mm], df.loc[mm, c])
            if res:
                out.append({"score": c, "scope": et, "n_forged": int((df.loc[mm, "edit_type"] == et).sum()),
                            "n_genuine": int((df.loc[mm, "label"] == "genuine").sum()), **res})
    return pd.DataFrame(out)


def _md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def roc_chart(df: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = (df["label"] == "forged").astype(int)
    cols = ["rules_score"] + [c for c in df.columns if c.startswith("det_")] + ["combined_score"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    ax.set_facecolor("#ffffff"); fig.patch.set_facecolor("#ffffff")
    ax.plot([0, 1], [0, 1], color="#9aa0a6", lw=1, ls="--", label="chance")
    for i, c in enumerate(cols):
        m = df[c].notna()
        if len(set(y[m])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[m], df.loc[m, c])
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2, label=f"{c.replace('det_', '')} (AUC {roc_auc_score(y[m], df.loc[m, c]):.2f})")
    ax.set_xlabel("False-positive rate (genuine marksheets flagged)", color="#3c4043")
    ax.set_ylabel("True-positive rate (forged marksheets caught)", color="#3c4043")
    ax.set_title("Detector comparison on this dataset", loc="left", color="#202124", fontsize=11)
    ax.grid(True, color="#eceff1", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdc1c6")
    ax.tick_params(colors="#5f6368")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_outputs(df: pd.DataFrame, out_dir: Path, cfg: dict):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    met = compute(df)
    met.to_csv(out_dir / "metrics.csv", index=False)
    roc_chart(df, out_dir / "roc.png")

    flagged = df.assign(flag=df.band.isin(["review", "high"]))
    gen = flagged[flagged.label == "genuine"].groupby("carrier").flag.agg(["mean", "count"]).reset_index()
    gen.columns = ["carrier", "share_flagged", "n"]
    frg = flagged[flagged.label == "forged"].groupby("edit_type").flag.agg(["mean", "count"]).reset_index()
    frg.columns = ["edit_type", "share_flagged", "n"]

    allm = met[met.scope == "all"][["score", "auc", "tpr@fpr5%", "tpr@fpr1%", "n_forged", "n_genuine"]]
    per = met[met.scope != "all"].pivot(index="scope", columns="score", values="auc").reset_index()
    unavailable = df.attrs.get("unavailable", {})
    lines = [
        "# Marksheet forensics evaluation", "",
        f"Files analysed: {len(df)} ({int((df.label == 'forged').sum())} forged, {int((df.label == 'genuine').sum())} genuine).", "",
        "> Read this carefully: on synthetic data these numbers show that the harness works, NOT how good a detector is on real "
        "Kerala marksheets. The synthetic forgeries are proxies. Add real AI-edited samples before drawing conclusions.", "",
        "## Overall (AUC and true-positive rate at a fixed false-positive rate)", "", _md(allm), "",
        "## AUC by forgery type (each type vs all genuine files)", "", _md(per), "",
        f"## Routing at current thresholds (review >= {cfg['thresholds']['review']}, high >= {cfg['thresholds']['high']})", "",
        "Genuine files sent to a human (false-alarm rate) by capture type:", "", _md(gen), "",
        "Forged files sent to a human (catch rate) by forgery type:", "", _md(frg), "",
    ]
    if unavailable:
        lines += ["## Detectors that did not run", ""] + [f"- **{k}**: {v}" for k, v in unavailable.items()] + [""]
    lines += ["![ROC](roc.png)", ""]
    (out_dir / "report.md").write_text("\n".join(lines))
    return met

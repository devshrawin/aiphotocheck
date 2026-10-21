# Marksheet forensics: evaluation harness

Compares ways of spotting tampered or AI-edited marksheets on the same files, so you choose detectors
with evidence instead of vendor claims. Built for the Unified Registry attachment-verification idea.

## Quick start (macOS)

```bash
./setup.sh                 # brew deps, venv, Python packages, ML packages, HF weights, TruFor clone
source .venv/bin/activate
python -m mf doctor        # shows which detectors are ready and why others are not
./run_eval.sh              # generates synthetic samples, evaluates, writes results/report.md
```

`WITH_ML=0 ./setup.sh` installs only the rules layer + classical baseline (no torch).

Other commands:

```bash
python -m mf make-samples --out data --n 12
python -m mf scan some_marksheet.pdf --expect '{"name":"A B","roll":"123456","dob":"01-02-2007"}'
python -m mf eval --data /path/to/labelled_folder --detectors classical,community_forensics
```

## What is in it

| Layer | File | What it does |
|---|---|---|
| Rules | `mf/rules.py` | File type vs extension, PDF revisions, producer/editor tools, modified-after-created, font-subset mixing, extra images on a page, EXIF editor tags, AI/provenance markers (C2PA etc.) |
| Content | `mf/marksheet.py` | OCR (two passes) or PDF text layer, subject totals and percentage arithmetic, optional registry cross-check (name, roll no., DOB) |
| Detectors | `mf/detectors/` | `classical` (ELA + noise residual, always on), `community_forensics` and `sdxl_detector` (Hugging Face whole-image classifiers), `trufor` (localisation, external repo), `seed` (document tamper localisation) |
| Combine | `mf/pipeline.py` | Noisy-OR of rule findings + weighted detector score gives a band: low / review / high |
| Evaluate | `mf/evaluate.py` | AUC, TPR at 1% and 5% false-positive rate, per forgery type, per capture type, ROC chart |

## Product rules baked in

- The **citizen** only gets `citizen_message`: no score, no "AI", no accusation. They are told to upload a clearer copy or fetch the document from the issuer.
- The **officer** gets `officer_view`: band, plain-language reasons, per-detector scores, heat maps.
- A band only routes a case to human review. It never rejects anyone.
- Scores are **not calibrated probabilities**. Never show "83% altered" to anyone.

## Dataset layout for your own evaluation

```
data/labels.csv      path,label,edit_type,carrier,sample_id    (label = genuine | forged)
data/registry.json   {"<sample_id>": {"name": "...", "roll": "...", "dob": "dd-mm-yyyy"}}   (optional)
data/genuine/...     real marksheets: scans, phone photos, PDFs
data/forged/...      tampered copies, including REAL AI-edited ones (edit_type = ai_inpainted)
```

Use real marksheets only with proper authority and data-protection approval, and keep them on approved infrastructure.

## Honest status of each part

| Part | Status |
|---|---|
| Rules, OCR/arithmetic checks, classical detector, synthetic generator, evaluator, CLI | Built and tested (10 unit tests, end-to-end run on 36 synthetic files) |
| `community_forensics`, `sdxl_detector` adapters | Written, **not run**: the build sandbox had no PyTorch/Hugging Face access. Model IDs come from the hub; if one fails to load, `doctor` says why. |
| `trufor` adapter | Written, **not run**. The command template in `config.json` is a best guess; check it against the repo's README. Needs weights downloaded manually. |
| `seed` adapter | Written, **not run**. Needs its GitHub repo cloned into `third_party/SEED`; input preprocessing is an assumption to verify. |

## What the synthetic run showed (36 files)

- Rules caught inconsistent edits, copy-move, name swaps and PDF text edits; no genuine file was flagged.
- Edits that keep the arithmetic consistent and then go through a scan/phone capture (`edit_consistent`, `inpaint_sim`) were **invisible** to the rules and the classical detector. That is the gap the ML detectors have to close, and it is the number to watch once they run.
- The classical detector scored at chance (AUC 0.47) on re-captured images, so it drags the combined score down at its current weight. Tune `detector_weights` (or fit a small logistic regression) on real data.
- `inpaint_sim` is a **proxy**, not a real generative-AI edit. Add real AI-edited samples before trusting any conclusion.

## Licences to clear before production

- **TruFor**: informational / non-profit use only.
- **Organika/sdxl-detector** and several community checkpoints: CC BY-NC (non-commercial).
- **Community Forensics** dataset is CC BY-NC-SA; check the model checkpoint's own licence.
- **SEED**: MIT for the solution code; base model and detector keep their upstream licences.

## Data protection

Run everything on approved infrastructure. Do not send citizen documents to public detector APIs or Hugging Face Spaces.
Weights are downloaded once (setup) and inference runs locally.

## Not yet covered

Issuer verification (DigiLocker / board QR / signature checks), per-board OCR templates, Malayalam OCR (`tesseract-lang`),
a learned score combiner, and the officer review UI.

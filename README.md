# aicheck: was this document, picture or attachment generated or altered with AI?

Give it an image, PDF or Office file (docx, pptx, xlsx). It returns an AI-likelihood score, a LOW / REVIEW / HIGH band,
plain-language reasons, and heat maps for a human reviewer. Built for the Kerala Unified Registry scheme-scrutiny use case,
but nothing in it is specific to one document type.

It is decision support. A band routes a file to a human; it never rejects anyone.

## Setup (macOS)

```bash
./setup.sh                    # brew poppler, venv, Python packages, ML packages, Hugging Face weights, TruFor clone
source .venv/bin/activate
python -m aicheck doctor      # shows which detectors are ready and why others are not
```

`WITH_ML=0 ./setup.sh` installs only the provenance checks and the classical detector (no torch).

## Use it

Check one file, several files, or whole folders:

```bash
python -m aicheck scan photo.jpg
python -m aicheck scan application.pdf attachments/ --out results
python -m aicheck scan photo.jpg --json          # machine-readable output for another system
python -m aicheck scan photo.jpg --deep          # also run TruFor and SEED: slow (minutes per image on CPU)
```

Example output:

```
photo.png
  AI likelihood: 97.8% (uncalibrated)   band: HIGH   type: png
   - PNG text chunk contains generation parameters (prompt/sampler/seed), typical of Stable Diffusion / ComfyUI output.
   heatmap: results/work/photo_classical_heatmap.png
```

Each scan also saves a JSON report in `results/reports/`.

## Make the percentage mean something (calibrate)

Out of the box the percentage is an **uncalibrated heuristic** and says so. To turn it into a measured number, give it
two folders of files you know the answer for:

```bash
python -m aicheck calibrate --real path/to/real_files --ai path/to/ai_files
cp results_calibration/calibration.json calibration.json     # scans pick this up automatically
```

This prints AUC and the catch rate at 1% and 5% false-alarm rates for each detector, and fits a logistic regression so
the percentage matches observed frequencies. Use at least 50 files per folder, ideally 200+, from the same sources your
officers will actually see (phone photos, scans, screenshots, WhatsApp-compressed images, recompressed AI images).

## What it checks

| Layer | File | What it does |
|---|---|---|
| Files | `aicheck/files.py` | Identifies the file type from its bytes, pulls images out of PDFs and Office files |
| Provenance | `aicheck/provenance.py` | C2PA / IPTC "trained algorithmic media" tags, generator strings, Stable Diffusion / ComfyUI prompt chunks in PNGs, EXIF editor tags, PDF revisions, editor in PDF producer, font mixing, extension mismatch. Precise when present, silent when metadata is stripped |
| Detectors | `aicheck/detectors/` | `classical` (ELA + noise residual), `community_forensics` and `sdxl_detector` (AI-image classifiers), `trufor` and `seed` (edit localisation, `--deep` only) |
| Scoring | `aicheck/pipeline.py` | Combines provenance evidence and detector scores (noisy-OR) into the percentage and band |
| Calibration | `aicheck/calibrate.py` | Measures detectors on labelled folders and fits the calibration |

Provenance findings are split into `ai` evidence (counts toward the percentage), `edit` evidence (non-AI manipulation,
listed separately and can raise a LOW file to REVIEW) and `info` (context only).

## Honest status

| Part | Status |
|---|---|
| Provenance checks, file handling, scoring, CLI | Built, 10 unit tests pass |
| `classical`, `community_forensics`, `sdxl_detector` | Run end to end. On 712 real receipts (SROIE) none scored above 0.5, so no false alarms there |
| Detector **sensitivity** | **Not measured.** No AI-generated documents have been tested yet. `community_forensics` scored every real receipt at about 0.40 and `sdxl_detector` about 0.00, so they may not be sensitive to documents at all. Do not rely on them until `calibrate` shows otherwise |
| `trufor`, `seed` | Run on one image each (TruFor ~110 s, SEED ~25 s on CPU). Not evaluated at scale |
| Calibration | None shipped. Percentages are marked uncalibrated until you run `calibrate` |

AI-edited documents (inpainted fields) are the hardest case. Published benchmarks show current detectors, including TruFor,
perform modestly on them. Expect the strongest signal from provenance metadata, which survives only when the file has not
been screenshotted or re-saved by a messaging app.

## Licences to clear before production

- **TruFor**: informational / non-profit use only.
- **Organika/sdxl-detector** and several community checkpoints: CC BY-NC (non-commercial).
- **Community Forensics** dataset is CC BY-NC-SA; check the model checkpoint's own licence.
- **SEED**: MIT for the solution code; base model and detector keep their upstream licences.

## Data protection

Run everything on approved infrastructure. Do not send citizen documents to public detector APIs or Hugging Face Spaces.
Weights are downloaded once (setup) and inference runs locally.

## Not yet covered

Issuer verification (DigiLocker, QR, digital signatures), video and audio, HEIC photos, a learned score combiner trained on
real data, and the officer review UI.

## Tests

```bash
python -m pytest tests -q
```

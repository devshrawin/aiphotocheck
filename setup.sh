#!/usr/bin/env bash
# One-time setup for macOS (Apple Silicon or Intel).
#   ./setup.sh            core + ML detectors
#   WITH_ML=0 ./setup.sh  core only (provenance checks + classical detector)
set -euo pipefail
cd "$(dirname "$0")"

echo "==> System tools (poppler, for PDF pages)"
if command -v brew >/dev/null 2>&1; then
  for pkg in poppler; do
    brew list "$pkg" >/dev/null 2>&1 || brew install "$pkg"
  done
else
  echo "Homebrew not found. Install it from https://brew.sh, then re-run, or install poppler yourself."
  exit 1
fi

echo "==> Python virtual environment"
PY="${PYTHON:-$(command -v python3.12 || command -v python3)}"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

if [ "${WITH_ML:-1}" = "1" ]; then
  echo "==> ML dependencies (torch, transformers, ...)"
  pip install -r requirements-ml.txt

  echo "==> Pre-downloading Hugging Face detector weights"
  python - <<'PY'
import json
from huggingface_hub import snapshot_download
cfg = json.load(open("config.json"))
for name, m in cfg["hf_models"].items():
    try:
        snapshot_download(m["model_id"])
        print("ok  ", name, m["model_id"])
    except Exception as e:
        print("FAIL", name, m["model_id"], "->", e)
try:
    snapshot_download(cfg["seed"]["hf_id"])
    print("ok   seed weights", cfg["seed"]["hf_id"])
except Exception as e:
    print("FAIL seed weights ->", e)
PY

  echo "==> Third-party repos"
  mkdir -p third_party
  if [ ! -d third_party/TruFor ]; then
    git clone https://github.com/grip-unina/TruFor third_party/TruFor
  fi
  cat <<'MSG'

Manual steps for the two localisation detectors:
  * TruFor: download the pretrained weights as described in third_party/TruFor/README.md and
    put them where that README says. NOTE: TruFor's licence allows informational / non-profit
    use only. Get commercial permission before using it in a government platform.
  * SEED: clone the GitHub repo linked from https://huggingface.co/Jason37437/SEED into
    third_party/SEED (the model card links it). Check its licence and the exact preprocessing.
MSG
fi

echo
echo "==> Done. Try:  source .venv/bin/activate && python -m aicheck doctor"

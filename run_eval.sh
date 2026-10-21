#!/usr/bin/env bash
# Usage: ./run_eval.sh [dataset_dir] [extra args for `python -m mf eval`]
#   No dataset_dir -> generates a synthetic sample set into ./data first.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

DATA="${1:-data}"
shift || true

if [ ! -f "$DATA/labels.csv" ]; then
  echo "==> No labels.csv in $DATA, generating synthetic samples"
  python -m mf make-samples --out "$DATA" --n 12
fi

python -m mf doctor
python -m mf eval --data "$DATA" --out results "$@"
echo "Report: results/report.md"

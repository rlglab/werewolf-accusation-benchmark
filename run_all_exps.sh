#!/usr/bin/env bash
# Recompute all three experiments from results/ into outputs/.
#
#   ./run_all_exps.sh [RESULTS_DIR] [OUT_DIR]
#
# Assumes the uv environment from the README's Setup section is activated.
set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RESULTS=${1:-"$SCRIPT_DIR/results"}
OUT=${2:-"$SCRIPT_DIR/outputs"}

python "$SCRIPT_DIR/code/exp1.py" --results "$RESULTS" --out-dir "$OUT"
python "$SCRIPT_DIR/code/exp2.py" --results "$RESULTS" --out-dir "$OUT"
python "$SCRIPT_DIR/code/exp3.py" --results "$RESULTS" --out-dir "$OUT"

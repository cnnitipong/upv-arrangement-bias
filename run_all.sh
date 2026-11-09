#!/usr/bin/env bash
# Re-runs the full pipeline from the CSV inputs in data/. Outputs go to outputs/.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export MPLBACKEND=Agg
mkdir -p outputs
for s in 01_build_master_and_core_stats 02_deep_analysis_and_grey_figures \
         03_extra_analyses 04_extra_figures \
         05_probabilistic_grading_and_repair_impact 06_confound_bounding \
         revision_analysis fix_figures_rineng; do
  echo "=== $s ==="
  "$PY" "src/$s.py" > "outputs/log_$s.txt" 2>&1 || { echo "FAILED: $s (see outputs/log_$s.txt)"; tail -20 "outputs/log_$s.txt"; exit 1; }
done
echo "All scripts finished. Outputs in outputs/"

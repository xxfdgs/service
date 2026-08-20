#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
TRAINING_CSV="${TRAINING_CSV:-datasets_lrx/raw/input/20260812-sum-700.csv}"
NEW_VALIDATION="${NEW_VALIDATION:-datasets_lrx/raw/feedback/new_validation.csv}"
MANIFEST_DIR="${MANIFEST_DIR:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests}"
OUTPUT_DIR="${OUTPUT_DIR:-results/simple_molecular_baselines/fifth_ood_v2}"

"$PYTHON" -u scripts/baseline/run_simple_molecular_baselines.py   --training-csv "$TRAINING_CSV"   --new-validation "$NEW_VALIDATION"   --manifest-dir "$MANIFEST_DIR"   --output-dir "$OUTPUT_DIR"   --targets Norm_before Norm_after   --seeds 100 101 102 103 104 105 106 107 108 109   --threshold 1.0

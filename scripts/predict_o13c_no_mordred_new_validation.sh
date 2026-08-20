#!/usr/bin/env bash
# Frozen O13-C mean-pooling, no-Mordred ensemble on labelled new_validation.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-results/input_graphgps_optimization/o13f_semantic_features}"
NEW_VALIDATION="${NEW_VALIDATION:-datasets_lrx/raw/feedback/new_validation.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MODEL_ROOT/new_validation_ensemble}"

[[ -f "$NEW_VALIDATION" ]] || { echo "Missing new_validation: $NEW_VALIDATION" >&2; exit 2; }
for group in core4 norm2; do
    "$PYTHON" scripts/diagnostics/predict_o12_10seed_ensemble.py \
        --model-root "$MODEL_ROOT" --input-files "$NEW_VALIDATION" \
        --output-root "$OUTPUT_ROOT" --target-group "$group" \
        --first-seed 100 --seed-count 10
    "$PYTHON" scripts/diagnostics/score_labelled_o12_10seed_ensemble.py \
        --labels-csv "$NEW_VALIDATION" --prediction-dir "$OUTPUT_ROOT/new_validation" \
        --output-dir "$OUTPUT_ROOT/new_validation/scored_${group}" --target-group "$group" \
        --model-label "O13-C mean pooling, no Mordred11"
done

echo "Finished: $OUTPUT_ROOT/new_validation/scored_core4 and scored_norm2"

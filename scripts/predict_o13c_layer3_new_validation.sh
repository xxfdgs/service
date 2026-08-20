#!/usr/bin/env bash
# Frozen three-layer O13-C mean-pooling ensemble on labelled new_validation.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-results/input_graphgps_optimization/o13c_mean_graph_pooling_fifth_identity_ood_seed100_109_no_mordred_no_aux}"
NEW_VALIDATION="${NEW_VALIDATION:-datasets_lrx/raw/feedback/new_validation.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MODEL_ROOT/new_validation_ensemble}"
SPLIT_SEEDS=(100 101 102 103 104 105 106 107 108 109)

[[ -f "$NEW_VALIDATION" ]] || { echo "Missing new_validation: $NEW_VALIDATION" >&2; exit 2; }
for group in core4 norm2; do
    "$PYTHON" scripts/diagnostics/predict_o12_10seed_ensemble.py \
        --model-root "$MODEL_ROOT" --input-files "$NEW_VALIDATION" \
        --output-root "$OUTPUT_ROOT" --target-group "$group" \
        --first-seed 100 --seed-count 10
    "$PYTHON" scripts/diagnostics/score_labelled_o12_10seed_ensemble.py \
        --labels-csv "$NEW_VALIDATION" --prediction-dir "$OUTPUT_ROOT/new_validation" \
        --output-dir "$OUTPUT_ROOT/new_validation/scored_${group}" --target-group "$group" \
        --model-label "O13-C mean pooling, 3 GraphGPS layers"
done

echo "Finished: $OUTPUT_ROOT/new_validation/scored_core4 and scored_norm2"

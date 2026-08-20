#!/usr/bin/env bash
# Frozen O13-E ten-checkpoint ensemble inference and labelled scatter plots.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-results/input_graphgps_optimization/o13e_mean_pooling_fifth_descriptors/o13e_strict_train_only_scaling}"
PREPROCESSING_ROOT="${PREPROCESSING_ROOT:-results/input_graphgps_optimization/o13e_mean_pooling_fifth_descriptors/preprocessing}"
NEW_VALIDATION="${NEW_VALIDATION:-datasets_lrx/raw/feedback/new_validation.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/input_graphgps_optimization/o13e_mean_pooling_fifth_descriptors/new_validation_ensemble}"
SEEDS=(100 101 102 103 104 105 106 107 108 109)

[[ -f "$NEW_VALIDATION" ]] || { echo "Missing new_validation: $NEW_VALIDATION" >&2; exit 2; }
for group in core4 norm2; do
    "$PYTHON" scripts/diagnostics/predict_o13e_new_validation_ensemble.py \
        --model-root "$MODEL_ROOT" --preprocessing-root "$PREPROCESSING_ROOT" \
        --input-csv "$NEW_VALIDATION" --output-root "$OUTPUT_ROOT" \
        --target-group "$group" --seeds "${SEEDS[@]}"
    "$PYTHON" scripts/diagnostics/score_labelled_o12_10seed_ensemble.py \
        --labels-csv "$NEW_VALIDATION" --prediction-dir "$OUTPUT_ROOT/new_validation" \
        --output-dir "$OUTPUT_ROOT/new_validation/scored_${group}" --target-group "$group" \
        --model-label "O13-E Fifth-OOD ensemble"
done

echo "Finished: $OUTPUT_ROOT/new_validation/scored_core4 and scored_norm2"

#!/usr/bin/env bash
# Infer the ten-checkpoint O12 trial-I ensemble on labelled new_validation.
# Core4 and norm2 are handled separately because they have different target
# scales.  Only a group with all selected-best checkpoints for seeds 100-109
# is run, so an unfinished norm2 run never prevents core4 prediction.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-results/input_graphgps_optimization/o13d_mean_pooling_fifth_class_random_seed100_109}"
NEW_VALIDATION="${NEW_VALIDATION:-datasets_lrx/raw/feedback/new_validation.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MODEL_ROOT/new_validation_ensemble}"
SPLIT_SEEDS=(100 101 102 103 104 105 106 107 108 109)

if [[ ! -f "$NEW_VALIDATION" ]]; then
    echo "new_validation CSV is missing: $NEW_VALIDATION" >&2
    exit 2
fi

group_is_complete() {
    local target_group="$1"
    local split_seed run_dir
    for split_seed in "${SPLIT_SEEDS[@]}"; do
        run_dir="$MODEL_ROOT/$target_group/O12_split${split_seed}"
        if [[ ! -f "$run_dir/checkpoints/selected_best.pt" || \
              ! -f "$run_dir/effective_config.yaml" || \
              ! -f "$run_dir/run_settings.json" ]]; then
            return 1
        fi
    done
}

infer_group() {
    local target_group="$1"
    local prediction_dir="$OUTPUT_ROOT/new_validation"
    local score_dir="$prediction_dir/scored_${target_group}"

    echo "Inferring trial-I ${target_group} on $NEW_VALIDATION"
    "$PYTHON" scripts/diagnostics/predict_o12_10seed_ensemble.py \
        --model-root "$MODEL_ROOT" \
        --input-files "$NEW_VALIDATION" \
        --output-root "$OUTPUT_ROOT" \
        --target-group "$target_group"

    # The predictor explicitly replaces labels with zero in its loader-only
    # CSV.  This scorer joins the original labels only after inference to
    # calculate MAE/R2/Pearson/Spearman and create one scatter plot per target.
    "$PYTHON" scripts/diagnostics/score_labelled_o12_10seed_ensemble.py \
        --labels-csv "$NEW_VALIDATION" \
        --prediction-dir "$prediction_dir" \
        --output-dir "$score_dir" \
        --target-group "$target_group"
}

completed=0
for target_group in core4 norm2; do
    if group_is_complete "$target_group"; then
        infer_group "$target_group"
        completed=1
    else
        echo "Skipping ${target_group}: one or more selected-best checkpoints (100-109) are incomplete."
    fi
done

if (( ! completed )); then
    echo "No complete trial-I checkpoint group was found under: $MODEL_ROOT" >&2
    exit 1
fi

echo "Finished. Metrics and scatter plots are under: $OUTPUT_ROOT/new_validation/"

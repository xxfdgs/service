#!/usr/bin/env bash
# Train a four-target O12 model on fixed split seeds 100-109. Targets are
# Aerosolization, Recovery, Norm_before, and Norm_after; each is z-scored
# using only that seed's outer-training rows.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
DATASET_VARIANT="${DATASET_VARIANT:-input_only}"
DATASET_ROOT="${DATASET_ROOT:-results/input_graphgps_optimization/later4_input_plus_feedback71}"
case "$DATASET_VARIANT" in
    input_only)
        DEFAULT_INPUT_CSV="$DATASET_ROOT/input_20260703_sum_utf8.csv"
        DEFAULT_MANIFESTS="results/input_graphgps_optimization/five_split_manifests"
        ;;
    input_plus_feedback71)
        DEFAULT_INPUT_CSV="$DATASET_ROOT/input_20260703_sum_plus_feedback71.csv"
        DEFAULT_MANIFESTS="$DATASET_ROOT/five_split_manifests_augmented"
        ;;
    *)
        echo "Unsupported DATASET_VARIANT=$DATASET_VARIANT; use input_only or input_plus_feedback71." >&2
        exit 2
        ;;
esac
INPUT_CSV="${INPUT_CSV:-$DEFAULT_INPUT_CSV}"
MANIFESTS="${MANIFESTS:-$DEFAULT_MANIFESTS}"
# Keep augmented-data checkpoints separate from input-only checkpoints.
RUNS_ROOT="${RUNS_ROOT:-results/input_graphgps_optimization/o12_later4_seed100_109_zscore_lr01_${DATASET_VARIANT}}"
BASE_CONFIG="${BASE_CONFIG:-results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/source_config.yaml}"
MORDRED="${MORDRED:-results/input_graphgps_optimization/features/mordred11_train_standardized.csv}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
# Stop after this many consecutive validation epochs without improvement.
# Override at launch, e.g. EARLY_STOP_PATIENCE=50 bash "$0".
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-200}"
# Architecture controls shared by all ten later4 runs.
GPS_LAYERS=2
GRAPH_HIDDEN_DIM=64
RWSE_DIM=16
FUSION_HIDDEN_DIM=64
HEAD_HIDDEN_DIM=64
VALIDATION_20260703="${VALIDATION_20260703:-datasets_lrx/raw/feedback/20260703_validation.csv}"
NEW_VALIDATION="${NEW_VALIDATION:-datasets_lrx/raw/feedback/new_validation.csv}"
INFERENCE_OUTPUT_ROOT="${INFERENCE_OUTPUT_ROOT:-$RUNS_ROOT/feedback_later4_ensemble}"
SPLIT_SEEDS=(100 101 102 103 104 105 106 107 108 109)

mkdir -p "$RUNS_ROOT/logs"

run_group() {
    local target_group="$1"
    local split_seed manifest run_dir candidate
    for split_seed in "${SPLIT_SEEDS[@]}"; do
        manifest="$MANIFESTS/split_manifest_seed${split_seed}.csv"
        run_dir="$RUNS_ROOT/O12_${target_group}_split${split_seed}"
        candidate="O12Later4_${target_group}_S${split_seed}"

        if [[ ! -f "$manifest" ]]; then
            echo "Missing fixed split manifest: $manifest" >&2
            exit 1
        fi
        if [[ -f "$run_dir/summary.json" && -f "$run_dir/predictions.csv" ]]; then
            echo "Skipping completed O12 later4 ${target_group} split ${split_seed}"
            continue
        fi

        resume_args=()
        if [[ -f "$run_dir/resume_state.pt" ]]; then
            resume_args+=(--resume)
        elif [[ -d "$run_dir" ]]; then
            resume_args+=(--restart-incomplete)
        fi

        echo "Training O12 later4 ${target_group} split ${split_seed}"
        "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
            --config "$BASE_CONFIG" \
            --run-dir "$run_dir" \
            --input-csv "$INPUT_CSV" \
            --target-set "$target_group" \
            --split-manifest "$manifest" \
            --fold "split${split_seed}" --group B --candidate "$candidate" \
            --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
            --seed 43 --base-lr 0.1 --weight-decay 1e-5 \
            --gt-dropout 0.1 --gt-attn-dropout 0.2 \
            --gps-layers "$GPS_LAYERS" --graph-hidden-dim "$GRAPH_HIDDEN_DIM" \
            --rwse-dim "$RWSE_DIM" --fusion-hidden-dim "$FUSION_HIDDEN_DIM" \
            --head-hidden-dim "$HEAD_HIDDEN_DIM" \
            --use-mordred-features --mordred-feature-dim 11 \
            --mordred-feature-path "$MORDRED" \
            --use-component-aux-features \
            --target-normalization zscore \
            --execution-max-epochs "$MAX_EPOCHS" \
            --early-stop-patience "$EARLY_STOP_PATIENCE" --include-test \
            "${resume_args[@]}" \
            > "$RUNS_ROOT/logs/O12_${target_group}_split${split_seed}.log" 2>&1
    done
}

run_group norm2
"$PYTHON" scripts/diagnostics/summarize_o12_later4_multitask.py \
    --runs-root "$RUNS_ROOT"

# Evaluate only the two labelled feedback tables after all checkpoints and the
# validation/test summary are complete. The predictor inverse-transforms each
# seed's train-only target scaler before averaging and plotting.
"$PYTHON" scripts/diagnostics/predict_o12_later4_feedback_ensemble.py \
    --model-root "$RUNS_ROOT" \
    --feedback-files "$VALIDATION_20260703" "$NEW_VALIDATION" \
    --output-root "$INFERENCE_OUTPUT_ROOT"

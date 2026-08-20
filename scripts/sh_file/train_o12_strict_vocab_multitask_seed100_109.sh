#!/usr/bin/env bash
# Train O12 core4 and norm2 multitask models on split seeds 100-109.  The
# current target keeps the historical vocabulary sizes [3, 4, 3, 4], including
# one unknown embedding row for each of the first four components.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-results/input_graphgps_optimization/later4_input_plus_feedback71}"
DATASET_VARIANT="${DATASET_VARIANT:-input_plus_feedback71}"
case "$DATASET_VARIANT" in
    input_plus_feedback71)
        DEFAULT_INPUT_CSV="$DATASET_ROOT/input_20260703_sum_plus_feedback71.csv"
        DEFAULT_MANIFESTS="$DATASET_ROOT/five_split_manifests_augmented"
        ;;
    feedback_only)
        DEFAULT_INPUT_CSV="$DATASET_ROOT/feedback_only_20260703_validation_nonoverlap.csv"
        DEFAULT_MANIFESTS="$DATASET_ROOT/five_split_manifests_feedback_only"
        ;;
    *)
        echo "Unsupported DATASET_VARIANT=$DATASET_VARIANT; use input_plus_feedback71 or feedback_only." >&2
        exit 2
        ;;
esac
INPUT_CSV="${INPUT_CSV:-$DEFAULT_INPUT_CSV}"
MANIFESTS="${MANIFESTS:-$DEFAULT_MANIFESTS}"
if [[ ! -f "$INPUT_CSV" ]]; then
    echo "Selected input CSV is missing: $INPUT_CSV" >&2
    exit 2
fi
if [[ ! -d "$MANIFESTS" ]]; then
    echo "Selected manifest directory is missing: $MANIFESTS" >&2
    exit 2
fi
# Keep sigmoid-core4 checkpoints separate from historical identity-readout
# runs. Core4 efficiencies are normalized to [0, 1] by the loader; sigmoid
# therefore bounds their reported predictions to the physical [0, 100] range.
# Norm2 retains its identity readout because it is not a percentage target.
RUNS_ROOT="${RUNS_ROOT:-results/input_graphgps_optimization/o12_multitask_seed100_109_lr001_${DATASET_VARIANT}_sigmoid_core4}"
BASE_CONFIG="${BASE_CONFIG:-results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/source_config.yaml}"
MORDRED="${MORDRED:-results/input_graphgps_optimization/features/mordred11_train_standardized.csv}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
# Stop after this many consecutive validation epochs without improvement.
# Override at launch, e.g. EARLY_STOP_PATIENCE=50 bash "$0".
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-200}"
# The current model keeps a 64-dimensional GraphGPS representation while the
# redesigned fusion and prediction head use a compact 32-dimensional space.
GPS_LAYERS=2
GRAPH_HIDDEN_DIM=64
RWSE_DIM=16
FUSION_HIDDEN_DIM=64
HEAD_HIDDEN_DIM=64
VALIDATION_20260703="${VALIDATION_20260703:-datasets_lrx/raw/feedback/20260703_validation.csv}"
NEW_VALIDATION="${NEW_VALIDATION:-datasets_lrx/raw/feedback/new_validation.csv}"
DOPE_PEPTIDE_PREDICT2="${DOPE_PEPTIDE_PREDICT2:-datasets_lrx/raw/feedback/20260723-DOPE-peptide-predict2.csv}"
LIBRARY_SINGLE_PREDICT="${LIBRARY_SINGLE_PREDICT:-datasets_lrx/raw/feedback/20260723-library-single-predict.csv}"
# ``INFERENCE_OUTPUT_ROOT`` remains a backward-compatible override for the
# norm2 output.  Core4 always has a separate output directory so the target
# columns, metrics, and scatter plots cannot overwrite norm2 artifacts.
NORM2_INFERENCE_OUTPUT_ROOT="${NORM2_INFERENCE_OUTPUT_ROOT:-${INFERENCE_OUTPUT_ROOT:-$RUNS_ROOT/feedback_norm2_ensemble}}"
CORE4_INFERENCE_OUTPUT_ROOT="${CORE4_INFERENCE_OUTPUT_ROOT:-$RUNS_ROOT/feedback_core4_ensemble}"
SPLIT_SEEDS=(100 101 102 103 104 105 106 107 108 109)

mkdir -p "$RUNS_ROOT/logs"

run_group() {
    local target_group="$1"
    local split_seed manifest run_dir candidate output_activation
    output_activation="identity"
    if [[ "$target_group" == "core4" ]]; then
        output_activation="sigmoid"
    fi
    for split_seed in "${SPLIT_SEEDS[@]}"; do
        manifest="$MANIFESTS/split_manifest_seed${split_seed}.csv"
        run_dir="$RUNS_ROOT/O12_${target_group}_split${split_seed}"
        candidate="O12Strict_${target_group}_S${split_seed}"

        if [[ ! -f "$manifest" ]]; then
            echo "Missing fixed split manifest: $manifest" >&2
            exit 1
        fi
        if [[ -f "$run_dir/summary.json" && -f "$run_dir/predictions.csv" ]]; then
            echo "Skipping completed O12 vocabulary-model ${target_group} split ${split_seed}"
            continue
        fi

        resume_args=()
        if [[ -f "$run_dir/resume_state.pt" ]]; then
            resume_args+=(--resume)
        elif [[ -d "$run_dir" ]]; then
            resume_args+=(--restart-incomplete)
        fi

        echo "Training O12 vocabulary-model ${target_group} split ${split_seed}"
        "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
            --config "$BASE_CONFIG" \
            --run-dir "$run_dir" \
            --input-csv "$INPUT_CSV" \
            --target-set "$target_group" \
            --split-manifest "$manifest" \
            --fold "split${split_seed}" --group B --candidate "$candidate" \
            --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
            --output-activation "$output_activation" \
            --seed 43 --base-lr 0.01 --weight-decay 1e-5 \
            --gt-dropout 0.1 --gt-attn-dropout 0.2 \
            --gps-layers "$GPS_LAYERS" --graph-hidden-dim "$GRAPH_HIDDEN_DIM" \
            --rwse-dim "$RWSE_DIM" --fusion-hidden-dim "$FUSION_HIDDEN_DIM" \
            --head-hidden-dim "$HEAD_HIDDEN_DIM" \
            --use-mordred-features --mordred-feature-dim 11 \
            --mordred-feature-path "$MORDRED" \
            --use-component-aux-features \
            --execution-max-epochs "$MAX_EPOCHS" \
            --early-stop-patience "$EARLY_STOP_PATIENCE" --include-test \
            "${resume_args[@]}" \
            > "$RUNS_ROOT/logs/O12_${target_group}_split${split_seed}.log" 2>&1
    done
}

# These two small models fit together on the 12-GB GPU.  Running one ordered
# seed stream per target group overlaps CPU graph batching without allowing
# same-output races.
#run_group core4 &
#core_pid=$!
#run_group norm2 &
#norm_pid=$!
#set +e
#wait "$core_pid"
#core_status=$?
#wait "$norm_pid"
#norm_status=$?
#set -e
#if (( core_status != 0 || norm_status != 0 )); then
#    echo "Training failed: core4 status=${core_status}, norm2 status=${norm_status}" >&2
#    exit 1
#fi

run_group core4
run_group norm2
"$PYTHON" scripts/diagnostics/summarize_o12_strict_vocab_multitask.py \
    --runs-root "$RUNS_ROOT"

# Evaluate both frozen target groups only after all training and the checkpoint
# summary above have succeeded.  For each group, labelled tables receive
# per-target MAE/R² CSVs plus true-vs-predicted scatter plots.
feedback_files=("$VALIDATION_20260703" "$NEW_VALIDATION")
if [[ "$DATASET_VARIANT" == "feedback_only" ]]; then
    # 20260703_validation contains the 71 feedback-only source rows, so it
    # would leak training labels into evaluation. new_validation is disjoint.
    feedback_files=("$NEW_VALIDATION")
fi
"$PYTHON" scripts/diagnostics/predict_o12_norm2_feedback_ensemble.py \
    --target-group core4 \
    --model-root "$RUNS_ROOT" \
    --feedback-files "${feedback_files[@]}" \
    --output-root "$CORE4_INFERENCE_OUTPUT_ROOT"
"$PYTHON" scripts/diagnostics/predict_o12_norm2_feedback_ensemble.py \
    --target-group norm2 \
    --model-root "$RUNS_ROOT" \
    --feedback-files "${feedback_files[@]}" \
    --output-root "$NORM2_INFERENCE_OUTPUT_ROOT"

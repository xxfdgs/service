#!/usr/bin/env bash
# Train the three Stage-5 models on the same ten Fifth-OOD splits with the
# same learning rate.
#
# Models
# ------
# P0_random : random Comp5GraphEncoder initialization
# P1_PT_D   : Stage-4 PT-D Comp5 initialization
# P2_PT_DF  : Stage-4 PT-DF Comp5 initialization
#
# Common protocol
# ---------------
# Fifth-OOD split manifests : 100 ... 109
# training RNG seed         : 43 (fixed for every split/model)
# rest-model LR             : 1e-3
# Comp5 LR                  : 1e-3
#
# IMPORTANT:
# No --comp5-lr is passed.  This intentionally uses the historical
# single-learning-rate GraphGym optimizer path for ALL THREE models.
# Therefore P1/P2 are exact full-fine-tuning controls at the same LR as P0.
#
# Existing completed runs are reused.  An incomplete run directory causes a
# hard failure instead of being silently overwritten.
#
# Run:
#   bash scripts/pretrain/stage6/run_p0_p1_p2_lr1e3_10split.sh
#
# Optional:
#   SPLIT_SEEDS="103 104 105 106 107 108 109" bash ...
#   MAX_EPOCHS=300 bash ...

set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON="${PYTHON:-$(which python)}"
RUNNER="${RUNNER:-scripts/diagnostics/run_fusion_head_experiment.py}"

# ---------------------------------------------------------------------------
# Locked Stage-5 inputs.
# ---------------------------------------------------------------------------
O12_BASELINE_ROOT="${O12_BASELINE_ROOT:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"

INPUT_CSV="${INPUT_CSV:-$O12_BASELINE_ROOT/staging/20260812-sum-700_utf8.csv}"
BASE_CONFIG="${BASE_CONFIG:-$O12_BASELINE_ROOT/core4/O12_split100/source_config.yaml}"

MORDRED="${MORDRED:-results/input_graphgps_optimization/features/mordred11_train_standardized.csv}"

OOD_MANIFESTS="${OOD_MANIFESTS:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests}"

STAGE4_ROOT="${STAGE4_ROOT:-results/fifth_pretraining/stage4_graphgps_pretraining}"
PT_D="${PT_D:-$STAGE4_ROOT/PT_D/checkpoints/best_comp5_encoder_state_dict.pt}"
PT_DF="${PT_DF:-$STAGE4_ROOT/PT_DF/checkpoints/best_comp5_encoder_state_dict.pt}"

TRANSFER_ROOT="${TRANSFER_ROOT:-results/fifth_pretraining/stage5_downstream_transfer}"

# ---------------------------------------------------------------------------
# Common training settings.
# ---------------------------------------------------------------------------
BASE_LR="${BASE_LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-50}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"

# IMPORTANT: split seed changes the frozen Fifth-OOD partition.
# Training RNG remains fixed, so this is a split-robustness study, not a
# 10-training-RNG study.
TRAIN_RNG_SEED="${TRAIN_RNG_SEED:-43}"
SPLIT_SEEDS="${SPLIT_SEEDS:-100 101 102 103 104 105 106 107 108 109}"

read -r -a SPLITS <<< "$SPLIT_SEEDS"

# ---------------------------------------------------------------------------
# Preflight.
# ---------------------------------------------------------------------------
for path in \
    "$RUNNER" \
    "$INPUT_CSV" \
    "$BASE_CONFIG" \
    "$MORDRED" \
    "$PT_D" \
    "$PT_DF"; do
    [[ -f "$path" ]] || {
        echo "Missing locked input: $path" >&2
        exit 2
    }
done

for split_seed in "${SPLITS[@]}"; do
    manifest="$OOD_MANIFESTS/fifth_identity_manifest_seed${split_seed}.csv"
    [[ -f "$manifest" ]] || {
        echo "Missing frozen Fifth-OOD manifest: $manifest" >&2
        exit 2
    }
done

"$PYTHON" - <<'PY'
from pathlib import Path

runner = Path("scripts/diagnostics/run_fusion_head_experiment.py")
text = runner.read_text(encoding="utf-8")

# P1/P2 require the strict Stage-4 transfer hook.
required = [
    "--comp5-pretrained-checkpoint",
    "--comp5-pretrain-label",
    "load_stage4_comp5_encoder",
    "comp5_initialization.json",
]
missing = [marker for marker in required if marker not in text]
if missing:
    raise SystemExit(
        "Stage-5 transfer runner is missing required hooks: "
        + ", ".join(missing)
    )

print("[preflight] Stage-5 transfer hooks: PASS")
PY

# ---------------------------------------------------------------------------
# Completion predicate.
# ---------------------------------------------------------------------------
run_is_complete() {
    local run_dir="$1"

    [[ -f "$run_dir/summary.json" ]] &&
    [[ -f "$run_dir/predictions.csv" ]] &&
    [[ -f "$run_dir/checkpoints/selected_best.pt" ]] &&
    [[ -f "$run_dir/comp5_initialization.json" ]]
}

# ---------------------------------------------------------------------------
# Common argument vector.
# ---------------------------------------------------------------------------
run_common() {
    local split_seed="$1"
    local run_dir="$2"
    shift 2

    local manifest="$OOD_MANIFESTS/fifth_identity_manifest_seed${split_seed}.csv"

    "$PYTHON" -u "$RUNNER" \
        --config "$BASE_CONFIG" \
        --run-dir "$run_dir" \
        --input-csv "$INPUT_CSV" \
        --target-set norm2 \
        --single-target Norm_before \
        --split-manifest "$manifest" \
        --fold "fifth_identity_ood_split${split_seed}" \
        --group B \
        --candidate "Stage5NormBeforeFullFT" \
        --fusion-type concat_mlp \
        --head-type baseline \
        --model-type OneHotEmbedGPS \
        --graph-pooling mean \
        --use-fifth-class-embedding \
        --output-activation identity \
        --seed "$TRAIN_RNG_SEED" \
        --base-lr "$BASE_LR" \
        --weight-decay "$WEIGHT_DECAY" \
        --batch-size "$BATCH_SIZE" \
        --warmup-epochs "$WARMUP_EPOCHS" \
        --early-stop-patience "$EARLY_STOP_PATIENCE" \
        --gt-dropout 0.1 \
        --gt-attn-dropout 0.2 \
        --gps-layers 2 \
        --use-mordred-features \
        --mordred-feature-dim 11 \
        --mordred-feature-path "$MORDRED" \
        --use-component-aux-features \
        --execution-max-epochs "$MAX_EPOCHS" \
        --include-test \
        "$@"
}

run_model_split() {
    local label="$1"
    local split_seed="$2"
    local initialization="$3"

    local run_dir="$TRANSFER_ROOT/$label/split${split_seed}"
    local log_dir="$TRANSFER_ROOT/$label/logs"
    local log="$log_dir/split${split_seed}.log"

    mkdir -p "$log_dir"

    if [[ -e "$run_dir" ]]; then
        if run_is_complete "$run_dir"; then
            echo "[skip] completed: $label split${split_seed}"
            return
        fi

        echo "Refusing incomplete run directory:" >&2
        echo "  $run_dir" >&2
        echo "Delete it explicitly if this run should be restarted." >&2
        exit 1
    fi

    echo
    echo "================================================================================"
    echo "Model              : $label"
    echo "Fifth-OOD split    : $split_seed"
    echo "Training RNG seed  : $TRAIN_RNG_SEED"
    echo "Base/Comp5 LR      : $BASE_LR / $BASE_LR"
    echo "Initialization     : $initialization"
    echo "================================================================================"

    case "$label" in
        P0_random)
            run_common \
                "$split_seed" \
                "$run_dir" \
                --comp5-pretrain-label "P0_random" \
                2>&1 | tee "$log"
            ;;

        P1_PT_D)
            run_common \
                "$split_seed" \
                "$run_dir" \
                --comp5-pretrained-checkpoint "$PT_D" \
                --comp5-pretrain-label "P1_PT_D" \
                2>&1 | tee "$log"
            ;;

        P2_PT_DF)
            run_common \
                "$split_seed" \
                "$run_dir" \
                --comp5-pretrained-checkpoint "$PT_DF" \
                --comp5-pretrain-label "P2_PT_DF" \
                2>&1 | tee "$log"
            ;;

        *)
            echo "Unsupported model label: $label" >&2
            exit 2
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Run identical split sequence for all three models.
# Model-major ordering allows one model's 100-109 series to finish before the
# next. Existing 100-102 Stage-5 runs are automatically reused.
# ---------------------------------------------------------------------------
MODELS=(
    "P0_random"
    "P1_PT_D"
    "P2_PT_DF"
)

for label in "${MODELS[@]}"; do
    for split_seed in "${SPLITS[@]}"; do
        case "$label" in
            P0_random)
                initialization="random"
                ;;
            P1_PT_D)
                initialization="PT-D"
                ;;
            P2_PT_DF)
                initialization="PT-DF"
                ;;
        esac

        run_model_split \
            "$label" \
            "$split_seed" \
            "$initialization"
    done
done

echo
echo "================================================================================"
echo "P0/P1/P2 common-LR ten-split training complete."
echo "Models : ${MODELS[*]}"
echo "Splits : ${SPLITS[*]}"
echo "LR     : $BASE_LR for all trainable parameters"
echo "RNG    : $TRAIN_RNG_SEED fixed"
echo "Root   : $TRANSFER_ROOT"
echo "================================================================================"

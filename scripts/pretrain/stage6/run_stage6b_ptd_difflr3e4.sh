#!/usr/bin/env bash
# Stage 6B: PT-D differential learning rate, Comp5 LR = 3e-4.
#
# Only effective change relative to Stage-5 P1_PT_D:
#   rest of model LR = 1e-3
#   Comp5 encoder LR = 3e-4
#
# Default screening splits: 100 101 102.
#
# Output intentionally lives under the existing Stage-5 transfer root so the
# same new_validation inference script can later consume it as another model:
#   P1_PT_D_diffLR1e4/split100...
#
# Run:
#   bash scripts/pretrain/stage6/run_stage6b_ptd_difflr3e4.sh

set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON="${PYTHON:-$(which python)}"
RUNNER="${RUNNER:-scripts/diagnostics/run_fusion_head_experiment.py}"

O12_BASELINE_ROOT="${O12_BASELINE_ROOT:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
INPUT_CSV="${INPUT_CSV:-$O12_BASELINE_ROOT/staging/20260812-sum-700_utf8.csv}"
BASE_CONFIG="${BASE_CONFIG:-$O12_BASELINE_ROOT/core4/O12_split100/source_config.yaml}"
MORDRED="${MORDRED:-results/input_graphgps_optimization/features/mordred11_train_standardized.csv}"
OOD_MANIFESTS="${OOD_MANIFESTS:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests}"

STAGE4_ROOT="${STAGE4_ROOT:-results/fifth_pretraining/stage4_graphgps_pretraining}"
PT_D="${PT_D:-$STAGE4_ROOT/PT_D/checkpoints/best_comp5_encoder_state_dict.pt}"

TRANSFER_ROOT="${TRANSFER_ROOT:-results/fifth_pretraining/stage5_downstream_transfer}"
LABEL="${LABEL:-P1_PT_D_diffLR3e4}"

REST_LR="${REST_LR:-0.001}"
COMP5_LR="${COMP5_LR:-0.0003}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
SPLIT_SEEDS="${SPLIT_SEEDS:-100 101 102 103 104 105 106 107 108 109}"

read -r -a SPLITS <<< "$SPLIT_SEEDS"

for path in \
    "$RUNNER" \
    "$INPUT_CSV" \
    "$BASE_CONFIG" \
    "$MORDRED" \
    "$PT_D"; do
    [[ -f "$path" ]] || {
        echo "Missing locked Stage-6 input: $path" >&2
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
required = [
    "--comp5-pretrained-checkpoint",
    "--comp5-pretrain-label",
    "--comp5-lr",
    "optimizer_parameter_groups.json",
    "Differential LR audit PASS",
]
missing = [marker for marker in required if marker not in text]
if missing:
    raise SystemExit(
        "Stage-6 runner patch is missing: " + ", ".join(missing)
    )
print("[Stage6] runner differential-LR hook audit: PASS")
PY

run_one() {
    local split_seed="$1"

    local manifest="$OOD_MANIFESTS/fifth_identity_manifest_seed${split_seed}.csv"
    local run_dir="$TRANSFER_ROOT/$LABEL/split${split_seed}"
    local log_dir="$TRANSFER_ROOT/$LABEL/logs"
    local log="$log_dir/split${split_seed}.log"

    mkdir -p "$log_dir"

    if [[ -e "$run_dir" ]]; then
        if [[ -f "$run_dir/summary.json" && \
              -f "$run_dir/predictions.csv" && \
              -f "$run_dir/checkpoints/selected_best.pt" && \
              -f "$run_dir/comp5_initialization.json" && \
              -f "$run_dir/optimizer_parameter_groups.json" ]]; then
            echo "Skipping completed Stage6 $LABEL split $split_seed"
            return
        fi

        echo "Refusing incomplete Stage6 directory: $run_dir" >&2
        echo "Delete it explicitly if you intend a fresh restart." >&2
        exit 1
    fi

    echo
    echo "================================================================================"
    echo "Stage 6A | $LABEL | Fifth-OOD split $split_seed"
    echo "rest LR  = $REST_LR"
    echo "Comp5 LR = $COMP5_LR"
    echo "training RNG seed = 43"
    echo "================================================================================"

    # Candidate stays identical to Stage-5 P0/P1/P2, keeping model architecture
    # metadata invariant. Only optimizer parameter-group LR differs from P1.
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
        --seed 43 \
        --base-lr "$REST_LR" \
        --comp5-lr "$COMP5_LR" \
        --weight-decay 1e-5 \
        --batch-size 8 \
        --warmup-epochs 50 \
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
        --comp5-pretrained-checkpoint "$PT_D" \
        --comp5-pretrain-label "$LABEL" \
        2>&1 | tee "$log"
}

for split_seed in "${SPLITS[@]}"; do
    run_one "$split_seed"
done

echo
echo "Stage 6B screening completed."
echo "Results: $TRANSFER_ROOT/$LABEL"

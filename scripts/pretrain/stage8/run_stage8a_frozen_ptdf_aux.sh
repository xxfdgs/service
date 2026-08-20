#!/usr/bin/env bash
# Stage 8A — frozen PT-DF structural prior + random/trainable Fifth task branch.
#
# Architecture:
#   Fifth graph
#      ├── random/trainable Comp5GraphEncoder
#      └── frozen Stage-4 PT-DF Comp5GraphEncoder
#                 ↓
#          concatenate both embeddings
#                 ↓
#          historical O13-D fusion/context
#
# Screening protocol:
#   Norm_before
#   Fifth-OOD split100/101/102
#   training RNG seed 43
#   single LR = 1e-3 for every TRAINABLE parameter
#   frozen PT-DF branch has requires_grad=False and stays eval()
#
# Run:
#   bash scripts/pretrain/stage8/run_stage8a_frozen_ptdf_aux.sh

set -euo pipefail

cd "$(dirname "$0")/../../.."

# The project GraphGPS/RDKit stack is installed in this environment.  Keep
# PYTHON overridable for clusters, but do not silently fall back to a system
# interpreter that lacks the project dependencies.
PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
RUNNER="${RUNNER:-scripts/diagnostics/run_fusion_head_experiment.py}"

O12_BASELINE_ROOT="${O12_BASELINE_ROOT:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
INPUT_CSV="${INPUT_CSV:-$O12_BASELINE_ROOT/staging/20260812-sum-700_utf8.csv}"
BASE_CONFIG="${BASE_CONFIG:-$O12_BASELINE_ROOT/core4/O12_split100/source_config.yaml}"
MORDRED="${MORDRED:-results/input_graphgps_optimization/features/mordred11_train_standardized.csv}"
OOD_MANIFESTS="${OOD_MANIFESTS:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests}"

STAGE4_ROOT="${STAGE4_ROOT:-results/fifth_pretraining/stage4_graphgps_pretraining}"
PT_DF="${PT_DF:-$STAGE4_ROOT/PT_DF/checkpoints/best_comp5_encoder_state_dict.pt}"

TRANSFER_ROOT="${TRANSFER_ROOT:-results/fifth_pretraining/stage5_downstream_transfer}"
LABEL="${LABEL:-P3_PT_DF_FrozenAux}"

BASE_LR="${BASE_LR:-0.001}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
SPLIT_SEEDS="${SPLIT_SEEDS:-100 101 102}"
TRAIN_RNG_SEED="${TRAIN_RNG_SEED:-43}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

read -r -a SPLITS <<< "$SPLIT_SEEDS"

[[ -x "$PYTHON" ]] || {
    echo "Configured Python is not executable: $PYTHON" >&2
    echo "Set PYTHON to the interpreter containing PyTorch Geometric, RDKit, and GraphGPS." >&2
    exit 2
}

for path in \
    "$RUNNER" \
    "$INPUT_CSV" \
    "$BASE_CONFIG" \
    "$MORDRED" \
    "$PT_DF"; do
    [[ -f "$path" ]] || {
        echo "Missing locked Stage-8 input: $path" >&2
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
    "--frozen-comp5-aux-checkpoint",
    "frozen_comp5_aux_initialization.json",
    "frozen_comp5_aux_enable",
]
missing = [marker for marker in required if marker not in text]
if missing:
    raise SystemExit(
        "Stage-8 runner patch is missing: " + ", ".join(missing)
    )

model_file = Path("graphgps/network/onehot_embed_gps.py")
model_text = model_file.read_text(encoding="utf-8")
if "frozen_comp5_aux_encoder" not in model_text:
    raise SystemExit(f"Stage-8 model patch is missing: {model_file}")

print("[Stage8] architecture/runner preflight: PASS")
PY

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "[Stage8] PREFLIGHT_ONLY=1; training was not started."
    exit 0
fi

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
              -f "$run_dir/frozen_comp5_aux_initialization.json" ]]; then
            echo "[skip] completed: $LABEL split${split_seed}"
            return
        fi

        echo "Refusing incomplete Stage-8 directory:" >&2
        echo "  $run_dir" >&2
        echo "Delete it explicitly if a fresh restart is intended." >&2
        exit 1
    fi

    echo
    echo "================================================================================"
    echo "Stage 8A | $LABEL | Fifth-OOD split $split_seed"
    echo "task Comp5       : random + trainable"
    echo "structural prior : PT-DF + frozen"
    echo "trainable LR     : $BASE_LR"
    echo "training RNG     : $TRAIN_RNG_SEED"
    echo "================================================================================"

    "$PYTHON" -u "$RUNNER" \
        --config "$BASE_CONFIG" \
        --run-dir "$run_dir" \
        --input-csv "$INPUT_CSV" \
        --target-set norm2 \
        --single-target Norm_before \
        --split-manifest "$manifest" \
        --fold "fifth_identity_ood_split${split_seed}" \
        --group B \
        --candidate "Stage8FrozenPTDFAux" \
        --fusion-type concat_mlp \
        --head-type baseline \
        --model-type OneHotEmbedGPS \
        --graph-pooling mean \
        --use-fifth-class-embedding \
        --output-activation identity \
        --seed "$TRAIN_RNG_SEED" \
        --base-lr "$BASE_LR" \
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
        --comp5-pretrain-label "${LABEL}_task_random" \
        --frozen-comp5-aux-checkpoint "$PT_DF" \
        --frozen-comp5-aux-label "Stage4_PT_DF_frozen_structural_prior" \
        2>&1 | tee "$log"
}

for split_seed in "${SPLITS[@]}"; do
    run_one "$split_seed"
done

echo
echo "Stage 8A screening completed."
echo "Label : $LABEL"
echo "Splits: ${SPLITS[*]}"
echo "Root  : $TRANSFER_ROOT/$LABEL"

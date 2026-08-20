#!/usr/bin/env bash
# Stage 5: P0 random vs P1 PT-D vs P2 PT-DF
# Primary experiment: O13-D Full, single-target Norm_before,
# frozen Fifth-identity OOD manifests.
#
# Default is the 3-split screening run: 100 101 102.
# Expand later with:
#   SPLIT_SEEDS="100 101 102 103 104 105 106 107 108 109" bash ...

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
PT_DF="${PT_DF:-$STAGE4_ROOT/PT_DF/checkpoints/best_comp5_encoder_state_dict.pt}"

STAGE5_ROOT="${STAGE5_ROOT:-results/fifth_pretraining/stage5_downstream_transfer}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"

# These are SPLIT seeds.  Training RNG stays fixed at 43 to match O13-D.
SPLIT_SEEDS="${SPLIT_SEEDS:-100 101 102}"
read -r -a SPLITS <<< "$SPLIT_SEEDS"

for path in \
    "$RUNNER" \
    "$INPUT_CSV" \
    "$BASE_CONFIG" \
    "$MORDRED" \
    "$OOD_MANIFESTS/protocol.json" \
    "$PT_D" \
    "$PT_DF"; do
    [[ -f "$path" ]] || {
        echo "Missing locked Stage-5 input: $path" >&2
        exit 2
    }
done

"$PYTHON" - <<'PY'
from pathlib import Path
p = Path("scripts/diagnostics/run_fusion_head_experiment.py")
text = p.read_text()
required = [
    "--comp5-pretrained-checkpoint",
    "comp5_initialization.json",
    "load_stage4_comp5_encoder",
]
missing = [x for x in required if x not in text]
if missing:
    raise SystemExit(
        "Stage-5 runner patch is missing: " + ", ".join(missing)
    )
print("[Stage5] downstream transfer hook audit: PASS")
PY

mkdir -p "$STAGE5_ROOT"

run_one() {
    local split_seed="$1"
    local label="$2"
    local checkpoint="$3"

    local manifest="$OOD_MANIFESTS/fifth_identity_manifest_seed${split_seed}.csv"
    local root="$STAGE5_ROOT/$label"
    local run_dir="$root/split${split_seed}"
    local log="$root/logs/split${split_seed}.log"

    [[ -f "$manifest" ]] || {
        echo "Missing frozen Fifth-OOD manifest: $manifest" >&2
        exit 2
    }

    mkdir -p "$root/logs"

    if [[ -e "$run_dir" ]]; then
        if [[ -f "$run_dir/summary.json" && \
              -f "$run_dir/predictions.csv" && \
              -f "$run_dir/checkpoints/selected_best.pt" && \
              -f "$run_dir/comp5_initialization.json" ]]; then
            echo "Skipping completed Stage5 $label split $split_seed"
            return
        fi
        echo "Refusing incomplete Stage5 directory: $run_dir" >&2
        echo "Use a fresh directory; do not reuse a partially built PyG cache." >&2
        exit 1
    fi

    local -a init_args=(
        --comp5-pretrain-label "$label"
    )
    if [[ -n "$checkpoint" ]]; then
        init_args+=(
            --comp5-pretrained-checkpoint "$checkpoint"
        )
    fi

    echo
    echo "======================================================================"
    echo "Stage5 $label | Fifth-OOD split $split_seed | Norm_before"
    echo "training RNG seed = 43"
    echo "======================================================================"

    # Keep candidate/architecture_name identical across P0/P1/P2; only Comp5 initialization differs.
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
        --base-lr 0.001 \
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
        "${init_args[@]}" \
        2>&1 | tee "$log"
}

# Run paired by split: P0, P1, P2 on the exact same manifest.
for split_seed in "${SPLITS[@]}"; do
    run_one "$split_seed" "P0_random" ""
    run_one "$split_seed" "P1_PT_D" "$PT_D"
    run_one "$split_seed" "P2_PT_DF" "$PT_DF"
done

echo
echo "Stage-5 screening completed."
echo "Root: $STAGE5_ROOT"
echo "Splits: ${SPLITS[*]}"

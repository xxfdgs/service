#!/usr/bin/env bash
# Run the two frozen Stage-4 pretraining ablations against the current O13-D
# Fifth GraphGPS interface.

set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON="${PYTHON:-$(which python)}"

O12_BASELINE_ROOT="${O12_BASELINE_ROOT:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
BASE_CONFIG="${BASE_CONFIG:-$O12_BASELINE_ROOT/core4/O12_split100/source_config.yaml}"

LIBRARY="${LIBRARY:-results/fifth_pretraining/stage2c_pretraining_library/stage2c_pretraining_molecular_library.csv}"
STAGE3_DIR="${STAGE3_DIR:-results/fifth_pretraining/stage3_pretraining_targets}"
STAGE4_ROOT="${STAGE4_ROOT:-results/fifth_pretraining/stage4_graphgps_pretraining}"

RUNNER="scripts/pretrain/stage4/pretrain_stage4_graphgps.py"

for path in "$BASE_CONFIG" "$LIBRARY" \
            "$STAGE3_DIR/descriptor_targets_scaled.npz" \
            "$STAGE3_DIR/morgan_fp_1024.npz" \
            "$STAGE3_DIR/morgan_fp_train_statistics.npz" \
            "$STAGE3_DIR/pretraining_split.csv"; do
    [[ -f "$path" ]] || {
        echo "Missing Stage-4 input: $path" >&2
        exit 2
    }
done

mkdir -p "$STAGE4_ROOT"

run_one() {
    local task="$1"
    local name="$2"
    local run_dir="$STAGE4_ROOT/$name"

    [[ ! -e "$run_dir" ]] || {
        echo "Refusing existing Stage-4 run directory: $run_dir" >&2
        exit 2
    }

    "$PYTHON" "$RUNNER" \
        --config "$BASE_CONFIG" \
        --library "$LIBRARY" \
        --stage3-dir "$STAGE3_DIR" \
        --run-dir "$run_dir" \
        --task "$task" \
        --graph-pooling mean \
        --gps-layers 2 \
        --gt-dropout 0.1 \
        --gt-attn-dropout 0.2 \
        --batch-size 32 \
        --epochs 200 \
        --base-lr 0.001 \
        --weight-decay 1e-5 \
        --warmup-epochs 10 \
        --early-stop-patience 30 \
        --seed 43 \
        2>&1 | tee "$STAGE4_ROOT/${name}.log"
}

run_one descriptor_only PT_D
run_one descriptor_plus_morgan PT_DF

echo
echo "Stage-4 ablations completed."
echo "PT-D : $STAGE4_ROOT/PT_D"
echo "PT-DF: $STAGE4_ROOT/PT_DF"

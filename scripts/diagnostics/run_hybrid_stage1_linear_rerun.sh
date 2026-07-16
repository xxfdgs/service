#!/usr/bin/env bash
# Force-rerun every non-M3 linear Stage-1 shard after the inner-preprocessing
# cache correction. The terminal marker is the only completion signal accepted
# by the tree queue before candidate selection.
set -euo pipefail

ROOT="/home/puzexuan/study/code/blology_prediction/service"
PYTHON="/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python"
LOG_ROOT="/tmp/hybrid_stage1_logs"
MARKER="$ROOT/results/hybrid_embedding_tree_exp/stage1/fixed_preprocessor_linear_complete.marker"

mkdir -p "$LOG_ROOT"
if [[ -e "$MARKER" ]]; then
    echo "LINEAR_FIXED_RERUN_ALREADY_COMPLETE"
    exit 0
fi

run_group() {
    local fold="$1"
    local tag="$2"
    local families="$3"
    printf '%s LINEAR_FIXED_RERUN_STARTED fold=%s tag=%s\n' "$(date --iso-8601=seconds)" "$fold" "$tag"
    env MPLCONFIGDIR=/tmp/matplotlib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        "$PYTHON" "$ROOT/scripts/diagnostics/run_hybrid_embedding_tree_stage.py" \
        --stage stage1 --fold "$fold" --families "$families" --models M0,M1,M2,M6 \
        --tag "$tag" --tree-n-jobs 1 --force >"$LOG_ROOT/${tag}.log" 2>&1
    printf '%s LINEAR_FIXED_RERUN_FINISHED fold=%s tag=%s\n' "$(date --iso-8601=seconds)" "$fold" "$tag"
}

for fold in 0 4; do
    run_group "$fold" "f${fold}_g1_linear" A0,A1,A2,A3,A4
    run_group "$fold" "f${fold}_g2_linear" A5,A6,A7,B1,B2
    run_group "$fold" "f${fold}_g3_linear" B3,B4,B5,B6,B7
    run_group "$fold" "f${fold}_g4_linear" B8,B9,B10,B11
done
touch "$MARKER"
printf '%s LINEAR_FIXED_RERUN_COMPLETE\n' "$(date --iso-8601=seconds)"

#!/usr/bin/env bash
# Durable foreground controller for the corrected non-M3 M4/M5 grid.  The
# execution environment reaps unattended background children, so this script
# waits on every launched batch before its next ten-minute completion check.
set -euo pipefail

ROOT="/home/puzexuan/study/code/blology_prediction/service"
PYTHON="/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python"
RERUN_ROOT="$ROOT/results/hybrid_embedding_tree_exp/stage1/fixed_preprocessor_rerun"
LINEAR_RERUN_MARKER="$ROOT/results/hybrid_embedding_tree_exp/stage1/fixed_preprocessor_linear_complete.marker"
LOG_ROOT="/tmp/hybrid_stage1_logs"
INTERVAL_SECONDS=600
MAX_BATCH=4
families=(A0 A1 A2 A3 A4 A5 A6 A7 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11)
models=(M4 M5)

tag_for() {
    local fold="$1" family="$2" model="$3"
    printf 'f%s_%s_x_%s' "$fold" "$family" "${model,,}"
}

marker_for() {
    local fold="$1" family="$2" model="$3"
    printf '%s/%s.done' "$RERUN_ROOT" "$(tag_for "$fold" "$family" "$model")"
}

pending_count() {
    local count=0 fold family model
    for fold in 0 4; do
        for family in "${families[@]}"; do
            for model in "${models[@]}"; do
                [[ -f "$(marker_for "$fold" "$family" "$model")" ]] || ((count += 1))
            done
        done
    done
    printf '%s\n' "$count"
}

run_task() {
    local fold="$1" family="$2" model="$3"
    local tag marker
    tag="$(tag_for "$fold" "$family" "$model")"
    marker="$(marker_for "$fold" "$family" "$model")"
    env MPLCONFIGDIR=/tmp/matplotlib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        "$PYTHON" "$ROOT/scripts/diagnostics/run_hybrid_embedding_tree_stage.py" \
        --stage stage1 --fold "$fold" --families "$family" --models "$model" \
        --tag "$tag" --tree-n-jobs 1 --force >"$LOG_ROOT/${tag}.log" 2>&1
    touch "$marker"
}

mkdir -p "$RERUN_ROOT" "$LOG_ROOT"
printf '%s TREE_BATCH_QUEUE_STARTED interval=%ss batch=%s\n' "$(date --iso-8601=seconds)" "$INTERVAL_SECONDS" "$MAX_BATCH"
while true; do
    # The status/launch decision is intentionally only made once per ten
    # minutes, per the user's monitoring instruction.
    sleep "$INTERVAL_SECONDS"
    pending="$(pending_count)"
    printf '%s TREE_BATCH_STATUS m3=deferred_not_read pending=%s\n' "$(date --iso-8601=seconds)" "$pending"
    if (( pending == 0 )); then
        [[ -f "$LINEAR_RERUN_MARKER" ]] || { echo "WAITING_FOR_LINEAR_RERUN"; continue; }
        cd "$ROOT"
        "$PYTHON" scripts/diagnostics/consolidate_hybrid_execution_manifest.py
        "$PYTHON" scripts/diagnostics/select_hybrid_embedding_tree_candidates.py
        candidate_rows=$(awk 'END { print NR - 1 }' results/hybrid_embedding_tree_exp/stage1/selected_candidates.csv)
        if (( candidate_rows == 0 )); then
            "$PYTHON" scripts/diagnostics/finalize_hybrid_early_stop.py
            echo "EARLY_STOP_FINALIZED"
        else
            bash scripts/diagnostics/run_hybrid_stage2_after_selection.sh
        fi
        exit 0
    fi

    pids=()
    launched=0
    for fold in 0 4; do
        for family in "${families[@]}"; do
            for model in "${models[@]}"; do
                [[ -f "$(marker_for "$fold" "$family" "$model")" ]] && continue
                tag="$(tag_for "$fold" "$family" "$model")"
                printf '%s TREE_BATCH_STARTED %s\n' "$(date --iso-8601=seconds)" "$tag"
                run_task "$fold" "$family" "$model" &
                pids+=("$!")
                ((launched += 1))
                (( launched >= MAX_BATCH )) && break 3
            done
        done
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    printf '%s TREE_BATCH_FINISHED launched=%s\n' "$(date --iso-8601=seconds)" "$launched"
done

#!/usr/bin/env bash
# Resume the non-M3 Stage-1 CPU tree grid without polling more often than once
# every ten minutes. M3 runs independently and is deliberately not inspected
# or used here until the user later requests it.
set -euo pipefail

ROOT="/home/puzexuan/study/code/blology_prediction/service"
PYTHON="/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python"
STAGE_ROOT="$ROOT/results/hybrid_embedding_tree_exp/stage1/shards"
RERUN_ROOT="$ROOT/results/hybrid_embedding_tree_exp/stage1/fixed_preprocessor_rerun"
LINEAR_RERUN_MARKER="$ROOT/results/hybrid_embedding_tree_exp/stage1/fixed_preprocessor_linear_complete.marker"
LOG_ROOT="/tmp/hybrid_stage1_logs"
INTERVAL_SECONDS=600
MAX_CONCURRENT=4

families=(A0 A1 A2 A3 A4 A5 A6 A7 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11)
models=(M4 M5)

metric_path() {
    local fold="$1"
    local family="$2"
    local model="$3"
    local tag="f${fold}_${family}_x_${model,,}"
    printf '%s/%s/fold_%s_metrics.csv' "$STAGE_ROOT" "$tag" "$fold"
}

rerun_marker() {
    local fold="$1"
    local family="$2"
    local model="$3"
    local tag="f${fold}_${family}_x_${model,,}"
    printf '%s/%s.done' "$RERUN_ROOT" "$tag"
}

is_running() {
    local tag="$1"
    pgrep -f -- "run_hybrid_embedding_tree_stage.py.*--tag ${tag}" >/dev/null 2>&1
}

start_task() {
    local fold="$1"
    local family="$2"
    local model="$3"
    local tag="f${fold}_${family}_x_${model,,}"
    local marker
    marker="$(rerun_marker "$fold" "$family" "$model")"
    mkdir -p "$LOG_ROOT"
    mkdir -p "$RERUN_ROOT"
    (
        env MPLCONFIGDIR=/tmp/matplotlib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            "$PYTHON" "$ROOT/scripts/diagnostics/run_hybrid_embedding_tree_stage.py" \
            --stage stage1 --fold "$fold" --families "$family" --models "$model" \
            --tag "$tag" --tree-n-jobs 1 --force >"$LOG_ROOT/${tag}.log" 2>&1 && touch "$marker"
    ) &
    printf '%s STARTED %s\n' "$(date --iso-8601=seconds)" "$tag"
}

pending_tree_count() {
    local count=0
    for fold in 0 4; do
        for family in "${families[@]}"; do
            for model in "${models[@]}"; do
                [[ -f "$(rerun_marker "$fold" "$family" "$model")" ]] || ((count += 1))
            done
        done
    done
    printf '%s\n' "$count"
}

running_tree_count() {
    pgrep -fc -- 'run_hybrid_embedding_tree_stage.py.*--models M[45]' || true
}

launch_available_tasks() {
    local running
    running="$(running_tree_count)"
    for fold in 0 4; do
        for family in "${families[@]}"; do
            for model in "${models[@]}"; do
                local tag="f${fold}_${family}_x_${model,,}"
                [[ -f "$(rerun_marker "$fold" "$family" "$model")" ]] && continue
                is_running "$tag" && continue
                if (( running >= MAX_CONCURRENT )); then
                    return
                fi
                start_task "$fold" "$family" "$model"
                ((running += 1))
            done
        done
    done
}

echo "$(date --iso-8601=seconds) QUEUE_STARTED interval=${INTERVAL_SECONDS}s"
while true; do
    # The wait comes before the status probe to honour the ten-minute cadence.
    sleep "$INTERVAL_SECONDS"
    pending="$(pending_tree_count)"
    running="$(running_tree_count)"
    printf '%s STATUS m3=deferred_not_read m4_m5_pending=%s running_m4_m5=%s\n' \
        "$(date --iso-8601=seconds)" "$pending" "$running"
    if (( pending > 0 )); then
        launch_available_tasks
        continue
    fi
    if [[ ! -f "$LINEAR_RERUN_MARKER" ]]; then
        printf '%s WAITING_FOR_LINEAR_PREPROCESSOR_RERUN\n' "$(date --iso-8601=seconds)"
        continue
    fi

    echo "$(date --iso-8601=seconds) STAGE1_TREE_GRID_COMPLETE"
    cd "$ROOT"
    "$PYTHON" scripts/diagnostics/consolidate_hybrid_execution_manifest.py
    "$PYTHON" scripts/diagnostics/select_hybrid_embedding_tree_candidates.py
    candidate_rows=$(awk 'END { print NR - 1 }' results/hybrid_embedding_tree_exp/stage1/selected_candidates.csv)
    if (( candidate_rows == 0 )); then
        "$PYTHON" scripts/diagnostics/finalize_hybrid_early_stop.py
        echo "$(date --iso-8601=seconds) EARLY_STOP_FINALIZED"
    else
        echo "$(date --iso-8601=seconds) STAGE1_CANDIDATES_READY count=${candidate_rows}"
        bash scripts/diagnostics/run_hybrid_stage2_after_selection.sh
    fi
    exit 0
done

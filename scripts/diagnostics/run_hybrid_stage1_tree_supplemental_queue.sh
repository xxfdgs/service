#!/usr/bin/env bash
# A staggered companion to the original ten-minute tree queue. It shares the
# fixed-generation completion markers and raises total CPU work to at most four
# single-threaded M4/M5 shards without reading any M3 output.
set -euo pipefail

ROOT="/home/puzexuan/study/code/blology_prediction/service"
PYTHON="/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python"
RERUN_ROOT="$ROOT/results/hybrid_embedding_tree_exp/stage1/fixed_preprocessor_rerun"
LOG_ROOT="/tmp/hybrid_stage1_logs"
INITIAL_DELAY_SECONDS=605
INTERVAL_SECONDS=600
MAX_GLOBAL_CONCURRENT=4
families=(A0 A1 A2 A3 A4 A5 A6 A7 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11)
models=(M4 M5)

marker_for() {
    local fold="$1" family="$2" model="$3"
    printf '%s/f%s_%s_x_%s.done' "$RERUN_ROOT" "$fold" "$family" "${model,,}"
}

tag_for() {
    local fold="$1" family="$2" model="$3"
    printf 'f%s_%s_x_%s' "$fold" "$family" "${model,,}"
}

running_count() {
    pgrep -fc -- 'run_hybrid_embedding_tree_stage.py.*--models M[45]' || true
}

is_running() {
    pgrep -f -- "run_hybrid_embedding_tree_stage.py.*--tag $1" >/dev/null 2>&1
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

start_task() {
    local fold="$1" family="$2" model="$3"
    local tag marker
    tag="$(tag_for "$fold" "$family" "$model")"
    marker="$(marker_for "$fold" "$family" "$model")"
    mkdir -p "$RERUN_ROOT" "$LOG_ROOT"
    (
        env MPLCONFIGDIR=/tmp/matplotlib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            "$PYTHON" "$ROOT/scripts/diagnostics/run_hybrid_embedding_tree_stage.py" \
            --stage stage1 --fold "$fold" --families "$family" --models "$model" \
            --tag "$tag" --tree-n-jobs 1 --force >"$LOG_ROOT/${tag}.log" 2>&1 && touch "$marker"
    ) &
    printf '%s SUPPLEMENTAL_STARTED %s\n' "$(date --iso-8601=seconds)" "$tag"
}

launch_available() {
    local running fold family model tag
    running="$(running_count)"
    for fold in 0 4; do
        for family in "${families[@]}"; do
            for model in "${models[@]}"; do
                tag="$(tag_for "$fold" "$family" "$model")"
                [[ -f "$(marker_for "$fold" "$family" "$model")" ]] && continue
                is_running "$tag" && continue
                (( running < MAX_GLOBAL_CONCURRENT )) || return 0
                start_task "$fold" "$family" "$model"
                ((running += 1))
            done
        done
    done
}

printf '%s SUPPLEMENTAL_QUEUE_STARTED initial_delay=%ss interval=%ss\n' "$(date --iso-8601=seconds)" "$INITIAL_DELAY_SECONDS" "$INTERVAL_SECONDS"
sleep "$INITIAL_DELAY_SECONDS"
while true; do
    pending="$(pending_count)"
    running="$(running_count)"
    printf '%s SUPPLEMENTAL_STATUS m3=deferred_not_read pending=%s running_m4_m5=%s\n' "$(date --iso-8601=seconds)" "$pending" "$running"
    (( pending == 0 )) && exit 0
    launch_available
    sleep "$INTERVAL_SECONDS"
done

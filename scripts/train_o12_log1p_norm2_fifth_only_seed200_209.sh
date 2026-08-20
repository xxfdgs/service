#!/usr/bin/env bash
# Train grouped-OOD O12 models whose regression fusion uses component five.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
MANIFESTS="${MANIFESTS:-results/input_graphgps_optimization/fifth_group_split_manifests}"
RUNS_ROOT="${RUNS_ROOT:-results/input_graphgps_optimization/o12_log1p_norm2_fifth_only_10seed}"
CACHE_SOURCE_ROOT="${CACHE_SOURCE_ROOT:-results/input_graphgps_optimization/o12_log1p_norm2_fifth_group_class_10seed}"
BASE_CONFIG="${BASE_CONFIG:-results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/source_config.yaml}"
MORDRED="${MORDRED:-results/input_graphgps_optimization/features/mordred11_train_standardized.csv}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
SPLIT_SEEDS=(200 201 202 203 204 205 206 207 208 209)

mkdir -p "$RUNS_ROOT/logs"

train_seed() {
    local split_seed="$1"
    local manifest="$MANIFESTS/fifth_group_manifest_seed${split_seed}.csv"
    local run_dir="$RUNS_ROOT/O12FifthOnly_split${split_seed}"
    local resume_args=()
    local cache_args=()
    if [[ ! -f "$manifest" ]]; then
        echo "Missing fifth-group split manifest: $manifest" >&2
        return 1
    fi
    if [[ -f "$run_dir/summary.json" && -f "$run_dir/predictions.csv" \
          && -f "$run_dir/checkpoints/selected_best.pt" ]]; then
        echo "Skipping completed O12 fifth-only split ${split_seed}"
        return 0
    fi
    local source_processed="$CACHE_SOURCE_ROOT/O12GroupClass_split${split_seed}/cache/.cache/double_fusion-head-norm2-fifthgroupclass${split_seed}-B-O12GroupClass${split_seed}_seed_${split_seed}/subset/processed"
    local target_processed="$run_dir/cache/.cache/double_fusion-head-norm2-fifthgroupfifthonly${split_seed}-B-O12FifthOnly${split_seed}_seed_${split_seed}/subset/processed"
    if [[ -f "$source_processed/train.pt" ]]; then
        if [[ ! -f "$target_processed/train.pt" ]]; then
            mkdir -p "$target_processed"
            cp -al "$source_processed/." "$target_processed/"
        fi
        cache_args+=(--reuse-existing-cache)
    fi
    if [[ -f "$run_dir/resume_state.pt" ]]; then
        resume_args+=(--resume)
    elif [[ -d "$run_dir" ]]; then
        resume_args+=(--restart-incomplete)
    fi
    echo "Training O12 grouped fifth-only log1p Norm split ${split_seed}"
    "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
        --config "$BASE_CONFIG" \
        --run-dir "$run_dir" \
        --target-set norm2 \
        --split-manifest "$manifest" \
        --fold "fifthgroupfifthonly${split_seed}" --group B \
        --candidate "O12FifthOnly${split_seed}" \
        --fusion-type concat_mlp --head-type baseline \
        --model-type OneHotEmbedGPS \
        --seed "$split_seed" --batch-size 32 --base-lr 0.001 \
        --weight-decay 1e-5 --warmup-epochs 20 \
        --gt-dropout 0.1 --gt-attn-dropout 0.2 \
        --use-mordred-features --mordred-feature-dim 11 \
        --mordred-feature-path "$MORDRED" \
        --mordred-fifth-only \
        --use-component-aux-features \
        --use-fifth-class-embedding \
        --fifth-only-fusion \
        --target-transform log1p \
        --execution-max-epochs "$MAX_EPOCHS" --include-test \
        "${cache_args[@]}" \
        "${resume_args[@]}" \
        > "$RUNS_ROOT/logs/O12FifthOnly_split${split_seed}.log" 2>&1
}

status=0
pids=()
for split_seed in "${SPLIT_SEEDS[@]}"; do
    train_seed "$split_seed" &
    pids+=("$!")
    if (( ${#pids[@]} >= MAX_PARALLEL )); then
        if ! wait "${pids[0]}"; then
            status=1
        fi
        pids=("${pids[@]:1}")
    fi
done
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if (( status != 0 )); then
    echo "At least one fifth-only O12 run failed." >&2
    exit "$status"
fi

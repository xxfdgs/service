#!/usr/bin/env bash
# Train ten continuous log1p Norm O12/GraphGPS models using input-only splits.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
MANIFESTS="${MANIFESTS:-results/input_graphgps_optimization/five_split_manifests}"
RUNS_ROOT="${RUNS_ROOT:-results/input_graphgps_optimization/o12_log1p_norm2_graphgps_10seed}"
BASE_CONFIG="${BASE_CONFIG:-results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/source_config.yaml}"
MORDRED="${MORDRED:-results/input_graphgps_optimization/features/mordred11_train_standardized.csv}"
CACHE_SOURCE_ROOT="${CACHE_SOURCE_ROOT:-results/input_graphgps_optimization/O12-10-seeds-prediction-models/norm2}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
SPLIT_SEEDS=(100 101 102 103 104 105 106 107 108 109)

mkdir -p "$RUNS_ROOT/logs"

train_seed() {
    local split_seed="$1"
    local manifest="$MANIFESTS/split_manifest_seed${split_seed}.csv"
    local run_dir="$RUNS_ROOT/O12Log_split${split_seed}"
    local resume_args=()
    local cache_args=()
    if [[ ! -f "$manifest" ]]; then
        echo "Missing fixed input-only split manifest: $manifest" >&2
        return 1
    fi
    if [[ -f "$run_dir/summary.json" && -f "$run_dir/predictions.csv" \
          && -f "$run_dir/checkpoints/selected_best.pt" ]]; then
        echo "Skipping completed O12 log1p Norm split ${split_seed}"
        return 0
    fi
    # The graph tensors depend on the input CSV, manifest and descriptors, not
    # on the target transform. Reuse the matching frozen O12 input cache to
    # avoid rebuilding the same 700 RDKit graphs for every new seed.
    local source_processed="$CACHE_SOURCE_ROOT/O12_split${split_seed}/cache/.cache/double_fusion-head-norm2-split${split_seed}-B-O12N${split_seed}_seed_43/subset/processed"
    local target_processed="$run_dir/cache/.cache/double_fusion-head-norm2-split${split_seed}-B-O12Log${split_seed}_seed_${split_seed}/subset/processed"
    if [[ ! -f "$target_processed/train.pt" && -d "$source_processed" ]]; then
        mkdir -p "$target_processed"
        cp -al "$source_processed/." "$target_processed/"
        cache_args+=(--reuse-existing-cache)
    elif [[ -f "$target_processed/train.pt" ]]; then
        cache_args+=(--reuse-existing-cache)
    fi
    # Cache prepopulation creates run_dir. Decide resume/restart only after
    # that step so a fresh run with a reused cache is not mistaken for an
    # unsafe overwrite attempt by the Python runner.
    if [[ -f "$run_dir/resume_state.pt" ]]; then
        resume_args+=(--resume)
    elif [[ -d "$run_dir" ]]; then
        resume_args+=(--restart-incomplete)
    fi
    echo "Training O12 log1p Norm split ${split_seed}"
    "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
        --config "$BASE_CONFIG" \
        --run-dir "$run_dir" \
        --target-set norm2 \
        --split-manifest "$manifest" \
        --fold "split${split_seed}" --group B --candidate "O12Log${split_seed}" \
        --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
        --seed "$split_seed" --batch-size 32 --base-lr 0.001 --weight-decay 1e-5 \
        --warmup-epochs 20 --gt-dropout 0.1 --gt-attn-dropout 0.2 \
        --use-mordred-features --mordred-feature-dim 11 \
        --mordred-feature-path "$MORDRED" \
        --use-component-aux-features \
        --target-transform log1p \
        --execution-max-epochs "$MAX_EPOCHS" --include-test \
        "${cache_args[@]}" \
        "${resume_args[@]}" \
        > "$RUNS_ROOT/logs/O12Log_split${split_seed}.log" 2>&1
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
    echo "At least one O12 log1p Norm run failed." >&2
    exit "$status"
fi

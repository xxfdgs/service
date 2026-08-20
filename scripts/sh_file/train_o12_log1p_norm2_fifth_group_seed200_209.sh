#!/usr/bin/env bash
# Train ten O12 log1p Norm models on fifth-identity-disjoint input splits.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
MANIFESTS="${MANIFESTS:-results/input_graphgps_optimization/fifth_group_split_manifests}"
RUNS_ROOT="${RUNS_ROOT:-results/input_graphgps_optimization/o12_log1p_norm2_fifth_group_10seed}"
BASE_CONFIG="${BASE_CONFIG:-results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/source_config.yaml}"
MORDRED="${MORDRED:-results/input_graphgps_optimization/features/mordred11_train_standardized.csv}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
SPLIT_SEEDS=(200 201 202 203 204 205 206 207 208 209)

mkdir -p "$RUNS_ROOT/logs"

train_seed() {
    local split_seed="$1"
    local manifest="$MANIFESTS/fifth_group_manifest_seed${split_seed}.csv"
    local run_dir="$RUNS_ROOT/O12Group_split${split_seed}"
    local resume_args=()
    if [[ ! -f "$manifest" ]]; then
        echo "Missing fifth-group split manifest: $manifest" >&2
        return 1
    fi
    if [[ -f "$run_dir/summary.json" && -f "$run_dir/predictions.csv" \
          && -f "$run_dir/checkpoints/selected_best.pt" ]]; then
        echo "Skipping completed O12 fifth-group split ${split_seed}"
        return 0
    fi
    if [[ -f "$run_dir/resume_state.pt" ]]; then
        resume_args+=(--resume)
    elif [[ -d "$run_dir" ]]; then
        resume_args+=(--restart-incomplete)
    fi
    echo "Training O12 log1p Norm fifth-group split ${split_seed}"
    "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
        --config "$BASE_CONFIG" \
        --run-dir "$run_dir" \
        --target-set norm2 \
        --split-manifest "$manifest" \
        --fold "fifthgroup${split_seed}" --group B \
        --candidate "O12Group${split_seed}" \
        --fusion-type concat_mlp --head-type baseline \
        --model-type OneHotEmbedGPS \
        --seed "$split_seed" --batch-size 32 --base-lr 0.001 \
        --weight-decay 1e-5 --warmup-epochs 20 \
        --gt-dropout 0.1 --gt-attn-dropout 0.2 \
        --use-mordred-features --mordred-feature-dim 11 \
        --mordred-feature-path "$MORDRED" \
        --use-component-aux-features \
        --target-transform log1p \
        --execution-max-epochs "$MAX_EPOCHS" --include-test \
        "${resume_args[@]}" \
        > "$RUNS_ROOT/logs/O12Group_split${split_seed}.log" 2>&1
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
    echo "At least one O12 fifth-group run failed." >&2
    exit "$status"
fi

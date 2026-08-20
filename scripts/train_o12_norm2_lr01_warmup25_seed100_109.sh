#!/usr/bin/env bash
# Fixed-lr O12 norm2 candidate: retain 64-D/2-layer encoder, shorten warmup.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
MANIFESTS="${MANIFESTS:-results/input_graphgps_optimization/five_split_manifests}"
RUNS_ROOT="${RUNS_ROOT:-results/input_graphgps_optimization/o12_norm2_lr01_dim64_l2_warmup25_seed100_109}"
BASE_CONFIG="${BASE_CONFIG:-results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/source_config.yaml}"
MORDRED="${MORDRED:-results/input_graphgps_optimization/features/mordred11_train_standardized.csv}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"

mkdir -p "$RUNS_ROOT/logs"

for split_seed in {100..109}; do
    manifest="$MANIFESTS/split_manifest_seed${split_seed}.csv"
    run_dir="$RUNS_ROOT/O12_norm2_split${split_seed}"
    log="$RUNS_ROOT/logs/O12_norm2_split${split_seed}.log"
    if [[ ! -f "$manifest" ]]; then
        echo "Missing split manifest: $manifest" >&2
        exit 1
    fi
    if [[ -f "$run_dir/checkpoints/selected_best.pt" && -f "$run_dir/predictions.csv" ]]; then
        echo "Skipping completed split${split_seed}"
        continue
    fi
    resume_args=()
    if [[ -f "$run_dir/resume_state.pt" ]]; then
        resume_args+=(--resume)
    elif [[ -d "$run_dir" ]]; then
        resume_args+=(--restart-incomplete)
    fi
    echo "Training split${split_seed}: lr=0.1, dim=64, gps_layers=2, warmup=25"
    "$PYTHON_BIN" scripts/diagnostics/run_fusion_head_experiment.py \
        --config "$BASE_CONFIG" --run-dir "$run_dir" --target-set norm2 \
        --split-manifest "$manifest" --fold "split${split_seed}" --group B \
        --candidate "O12LR01D64L2W25S${split_seed}" \
        --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
        --seed 43 --base-lr 0.1 --warmup-epochs 25 --weight-decay 1e-5 \
        --gps-layers 2 --graph-hidden-dim 64 --rwse-dim 28 \
        --gt-dropout 0.1 --gt-attn-dropout 0.2 \
        --use-mordred-features --mordred-feature-dim 11 \
        --mordred-feature-path "$MORDRED" --use-component-aux-features \
        --execution-max-epochs "$MAX_EPOCHS" --include-test "${resume_args[@]}" \
        > "$log" 2>&1
done

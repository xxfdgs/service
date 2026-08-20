#!/usr/bin/env bash
# Locked diagnostic benchmark: O12 on Fifth-identity-disjoint 80/10/10 splits.
# This reproduces the frozen baseline architecture/training configuration only;
# it intentionally adds no fifth-specific module, descriptor, or external data.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
BASELINE_ROOT="${BASELINE_ROOT:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
INPUT_CSV="${INPUT_CSV:-$BASELINE_ROOT/staging/20260812-sum-700_utf8.csv}"
MANIFESTS="${MANIFESTS:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests}"
BASE_CONFIG="${BASE_CONFIG:-$BASELINE_ROOT/core4/O12_split100/source_config.yaml}"
MORDRED="${MORDRED:-results/input_graphgps_optimization/features/mordred11_train_standardized.csv}"
RUNS_ROOT="${RUNS_ROOT:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
SPLIT_SEEDS=(100 101 102 103 104 105 106 107 108 109)
read -r -a TRAIN_TARGET_GROUPS <<< "${TRAIN_TARGET_GROUPS:-core4 norm2}"

for path in "$INPUT_CSV" "$BASE_CONFIG" "$MORDRED" "$MANIFESTS/protocol.json"; do
    if [[ ! -f "$path" ]]; then
        echo "Required locked-benchmark input is missing: $path" >&2
        exit 2
    fi
done
for target_group in "${TRAIN_TARGET_GROUPS[@]}"; do
    if [[ "$target_group" != "core4" && "$target_group" != "norm2" ]]; then
        echo "TRAIN_TARGET_GROUPS supports only core4 and norm2." >&2
        exit 2
    fi
done
mkdir -p "$RUNS_ROOT/logs"

run_group() {
    local target_group="$1"
    local candidate_prefix="$2"
    local output_activation="identity"
    if [[ "$target_group" == "core4" ]]; then
        output_activation="sigmoid"
    fi
    for split_seed in "${SPLIT_SEEDS[@]}"; do
        local manifest="$MANIFESTS/fifth_identity_manifest_seed${split_seed}.csv"
        local run_dir="$RUNS_ROOT/$target_group/O12_split${split_seed}"
        if [[ ! -f "$manifest" ]]; then
            echo "Missing Fifth-OOD manifest: $manifest" >&2
            exit 2
        fi
        if [[ -e "$run_dir" ]]; then
            if [[ -f "$run_dir/summary.json" && \
                  -f "$run_dir/predictions.csv" && \
                  -f "$run_dir/checkpoints/selected_best.pt" ]]; then
                echo "Skipping completed O12 Fifth-OOD ${target_group} split ${split_seed}"
                continue
            fi
            echo "Refusing to resume an incomplete Fifth-OOD run: $run_dir" >&2
            echo "Use a new RUNS_ROOT to rebuild its processed cache." >&2
            exit 1
        fi
        echo "Training locked O12 Fifth-OOD ${target_group} split ${split_seed}"
        "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
            --config "$BASE_CONFIG" --run-dir "$run_dir" --input-csv "$INPUT_CSV" \
            --target-set "$target_group" --split-manifest "$manifest" \
            --fold "fifth_ood_split${split_seed}" --group B --candidate "${candidate_prefix}${split_seed}" \
            --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
            --output-activation "$output_activation" --seed 43 \
            --base-lr 0.001 --weight-decay 1e-5 --batch-size 8 \
            --warmup-epochs 50 --early-stop-patience "$EARLY_STOP_PATIENCE" \
            --gt-dropout 0.1 --gt-attn-dropout 0.2 --gps-layers 2 \
            --use-mordred-features --mordred-feature-dim 11 --mordred-feature-path "$MORDRED" \
            --use-component-aux-features --execution-max-epochs "$MAX_EPOCHS" --include-test \
            2>&1 | tee "$RUNS_ROOT/logs/O12_${target_group}_split${split_seed}.log"
    done
}

for target_group in "${TRAIN_TARGET_GROUPS[@]}"; do
    case "$target_group" in
        core4) run_group core4 O12FifthOODCore4 ;;
        norm2) run_group norm2 O12FifthOODNorm2 ;;
    esac
done

for target_group in "${TRAIN_TARGET_GROUPS[@]}"; do
    for split_seed in "${SPLIT_SEEDS[@]}"; do
        if [[ ! -f "$RUNS_ROOT/$target_group/O12_split${split_seed}/checkpoints/selected_best.pt" ]]; then
            echo "Skipping evaluation: incomplete ${target_group} split ${split_seed}." >&2
            exit 0
        fi
    done
done

"$PYTHON" scripts/diagnostics/evaluate_o12_10seed_corresponding_splits.py \
    --model-root "$RUNS_ROOT" --manifest-root "$MANIFESTS" \
    --output-dir "$RUNS_ROOT/corresponding_split_single_inference" \
    --seeds "${SPLIT_SEEDS[@]}" --target-groups "${TRAIN_TARGET_GROUPS[@]}"

"$PYTHON" scripts/diagnostics/summarize_o12_fifth_identity_ood.py \
    --random-metrics "$BASELINE_ROOT/corresponding_split_single_inference/metrics_by_checkpoint_target.csv" \
    --ood-metrics "$RUNS_ROOT/corresponding_split_single_inference/metrics_by_checkpoint_target.csv" \
    --ood-predictions "$RUNS_ROOT/corresponding_split_single_inference/predictions_by_checkpoint.csv" \
    --output-dir "$RUNS_ROOT/benchmark_summary"

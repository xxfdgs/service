#!/usr/bin/env bash
# Part A engineering validation: one frozen Fifth-identity-OOD split, seed 100.
# O12 add and O13-C mean are both refit with the same train-only Mordred scaler.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
SEED="${SEED:-100}"
O12_ROOT="${O12_ROOT:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
INPUT_CSV="${INPUT_CSV:-$O12_ROOT/staging/20260812-sum-700_utf8.csv}"
BASE_CONFIG="${BASE_CONFIG:-$O12_ROOT/core4/O12_split100/source_config.yaml}"
MANIFEST_ROOT="${MANIFEST_ROOT:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests}"
MANIFEST="$MANIFEST_ROOT/fifth_identity_manifest_seed${SEED}.csv"
RAW_LOOKUP="${RAW_LOOKUP:-results/deduplicated_rebaseline/artifacts/mordred_11_lookup.csv}"
ROOT_OUT="${ROOT_OUT:-results/input_graphgps_optimization/o13_train_only_scaling_seed${SEED}_validation}"
LOOKUP="$ROOT_OUT/preprocessing/mordred11_train_only_seed${SEED}.csv"
O12_RUNS="$ROOT_OUT/o12_add_strict_scaling"
O13C_RUNS="$ROOT_OUT/o13c_mean_strict_scaling"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
TARGET_GROUPS=(core4 norm2)

for path in "$INPUT_CSV" "$BASE_CONFIG" "$MANIFEST" "$RAW_LOOKUP"; do
    [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done

"$PYTHON" scripts/diagnostics/build_o13_train_only_mordred11_lookup.py \
    --input-csv "$INPUT_CSV" --manifest "$MANIFEST" --raw-lookup "$RAW_LOOKUP" --output "$LOOKUP"

train_variant() {
    local variant="$1" pooling="$2" runs_root="$3"
    mkdir -p "$runs_root/logs"
    for target_group in "${TARGET_GROUPS[@]}"; do
        local output_activation="identity"
        [[ "$target_group" == "core4" ]] && output_activation="sigmoid"
        local run_dir="$runs_root/$target_group/O12_split${SEED}"
        if [[ -e "$run_dir" ]]; then
            if [[ -f "$run_dir/summary.json" && -f "$run_dir/predictions.csv" && \
                  -f "$run_dir/checkpoints/selected_best.pt" ]]; then
                echo "Skipping completed ${variant} ${target_group} seed ${SEED}"
                continue
            fi
            echo "Refusing incomplete run directory: $run_dir" >&2; exit 1
        fi
        "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
            --config "$BASE_CONFIG" --run-dir "$run_dir" --input-csv "$INPUT_CSV" \
            --target-set "$target_group" --split-manifest "$MANIFEST" \
            --fold "fifth_identity_ood_split${SEED}" --group B \
            --candidate "${variant}_${target_group}_seed${SEED}" \
            --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
            --graph-pooling "$pooling" --output-activation "$output_activation" \
            --seed 43 --base-lr 0.001 --weight-decay 1e-5 --batch-size 8 \
            --warmup-epochs 50 --early-stop-patience "$EARLY_STOP_PATIENCE" \
            --gt-dropout 0.1 --gt-attn-dropout 0.2 --gps-layers 2 \
            --use-mordred-features --mordred-feature-dim 11 --mordred-feature-path "$LOOKUP" \
            --use-component-aux-features --execution-max-epochs "$MAX_EPOCHS" --include-test \
            2>&1 | tee "$runs_root/logs/${variant}_${target_group}_seed${SEED}.log"
    done
    "$PYTHON" scripts/diagnostics/evaluate_o12_10seed_corresponding_splits.py \
        --model-root "$runs_root" --manifest-root "$MANIFEST_ROOT" \
        --output-dir "$runs_root/corresponding_split_single_inference" \
        --seeds "$SEED" --target-groups "${TARGET_GROUPS[@]}"
}

train_variant O12StrictScaling add "$O12_RUNS"
train_variant O13CStrictScaling mean "$O13C_RUNS"

for target_group in "${TARGET_GROUPS[@]}"; do
    "$PYTHON" scripts/diagnostics/audit_o13_train_only_scaling_pair.py \
        --o12-effective "$O12_RUNS/$target_group/O12_split${SEED}/effective_config.yaml" \
        --o13c-effective "$O13C_RUNS/$target_group/O12_split${SEED}/effective_config.yaml" \
        --strict-lookup "$LOOKUP" --scaler-metadata "${LOOKUP%.csv}.json" \
        --output "$ROOT_OUT/preprocessing/config_audit_${target_group}_seed${SEED}.json"
done

"$PYTHON" scripts/diagnostics/compare_o13_train_only_scaling_seed.py \
    --input-csv "$INPUT_CSV" \
    --o12-predictions "$O12_RUNS/corresponding_split_single_inference/predictions_by_checkpoint.csv" \
    --o13c-predictions "$O13C_RUNS/corresponding_split_single_inference/predictions_by_checkpoint.csv" \
    --output-dir "$ROOT_OUT/o12_vs_o13c_strict_scaling_comparison"

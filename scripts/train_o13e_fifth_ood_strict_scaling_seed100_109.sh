#!/usr/bin/env bash
# O13-E: strict Fifth-OOD O13-C reference versus O13-C + 12 fifth-only descriptors.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
O12_ROOT="${O12_ROOT:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
INPUT_CSV="${INPUT_CSV:-$O12_ROOT/staging/20260812-sum-700_utf8.csv}"
BASE_CONFIG="${BASE_CONFIG:-$O12_ROOT/core4/O12_split100/source_config.yaml}"
MANIFEST_ROOT="${MANIFEST_ROOT:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests}"
RAW_MORDRED="${RAW_MORDRED:-results/deduplicated_rebaseline/artifacts/mordred_11_lookup.csv}"
ROOT_OUT="${ROOT_OUT:-results/input_graphgps_optimization/o13e_mean_pooling_fifth_descriptors}"
RAW_FIFTH="$ROOT_OUT/raw_fifth_descriptor_lookup.csv"
O13C_ROOT="$ROOT_OUT/o13c_strict_train_only_scaling"
O13E_ROOT="$ROOT_OUT/o13e_strict_train_only_scaling"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
SPLIT_SEEDS=(100 101 102 103 104 105 106 107 108 109)
TARGET_GROUPS=(core4 norm2)

for path in "$INPUT_CSV" "$BASE_CONFIG" "$RAW_MORDRED" "$MANIFEST_ROOT/protocol.json"; do
    [[ -f "$path" ]] || { echo "Missing locked input: $path" >&2; exit 2; }
done
if [[ ! -f "$RAW_FIFTH" ]]; then
    PYTHONPATH=. "$PYTHON" scripts/diagnostics/build_o13e_raw_fifth_descriptor_lookup.py \
        --input-csv "$INPUT_CSV" --raw-mordred-lookup "$RAW_MORDRED" --output "$RAW_FIFTH"
fi

complete_run() {
    [[ -f "$1/summary.json" && -f "$1/predictions.csv" && -f "$1/checkpoints/selected_best.pt" ]]
}

for seed in "${SPLIT_SEEDS[@]}"; do
    manifest="$MANIFEST_ROOT/fifth_identity_manifest_seed${seed}.csv"
    [[ -f "$manifest" ]] || { echo "Missing frozen OOD manifest: $manifest" >&2; exit 2; }
    preprocess="$ROOT_OUT/preprocessing/seed${seed}"
    mordred_lookup="$preprocess/mordred11_all_components_train_only.csv"
    fifth_lookup="$preprocess/fifth_mechanistic_train_only.csv"
    mkdir -p "$preprocess"
    PYTHONPATH=. "$PYTHON" scripts/diagnostics/build_o13_train_only_mordred11_lookup.py \
        --input-csv "$INPUT_CSV" --manifest "$manifest" --raw-lookup "$RAW_MORDRED" --output "$mordred_lookup"
    PYTHONPATH=. "$PYTHON" scripts/diagnostics/build_o13e_train_only_fifth_lookup.py \
        --input-csv "$INPUT_CSV" --manifest "$manifest" --raw-fifth-lookup "$RAW_FIFTH" \
        --output "$fifth_lookup" --audit-dir "$preprocess/redundancy_audit"

    for target_group in "${TARGET_GROUPS[@]}"; do
        output_activation=identity
        [[ "$target_group" == core4 ]] && output_activation=sigmoid
        c_run="$O13C_ROOT/$target_group/O12_split${seed}"
        e_run="$O13E_ROOT/$target_group/O12_split${seed}"
        mkdir -p "$O13C_ROOT/logs" "$O13E_ROOT/logs"
        if complete_run "$c_run"; then
            echo "Skipping completed O13-C strict ${target_group} seed ${seed}"
        elif [[ -e "$c_run" ]]; then
            echo "Refusing incomplete O13-C strict run: $c_run" >&2; exit 1
        else
            PYTHONPATH=. "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
                --config "$BASE_CONFIG" --run-dir "$c_run" --input-csv "$INPUT_CSV" \
                --target-set "$target_group" --split-manifest "$manifest" --fold "fifth_identity_ood_split${seed}" \
                --group B --candidate "O13CStrictScaling_${target_group}_seed${seed}" \
                --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
                --graph-pooling mean --output-activation "$output_activation" --seed 43 --base-lr 0.001 \
                --weight-decay 1e-5 --batch-size 8 --warmup-epochs 50 --early-stop-patience "$EARLY_STOP_PATIENCE" \
                --gt-dropout 0.1 --gt-attn-dropout 0.2 --gps-layers 2 --use-mordred-features \
                --mordred-feature-dim 11 --mordred-feature-path "$mordred_lookup" --use-component-aux-features \
                --execution-max-epochs "$MAX_EPOCHS" --include-test 2>&1 | tee "$O13C_ROOT/logs/seed${seed}_${target_group}.log"
        fi
        if complete_run "$e_run"; then
            echo "Skipping completed O13-E strict ${target_group} seed ${seed}"
        elif [[ -e "$e_run" ]]; then
            echo "Refusing incomplete O13-E strict run: $e_run" >&2; exit 1
        else
            PYTHONPATH=. "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
                --config "$BASE_CONFIG" --run-dir "$e_run" --input-csv "$INPUT_CSV" \
                --target-set "$target_group" --split-manifest "$manifest" --fold "fifth_identity_ood_split${seed}" \
                --group B --candidate "O13E_${target_group}_seed${seed}" \
                --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
                --graph-pooling mean --output-activation "$output_activation" --seed 43 --base-lr 0.001 \
                --weight-decay 1e-5 --batch-size 8 --warmup-epochs 50 --early-stop-patience "$EARLY_STOP_PATIENCE" \
                --gt-dropout 0.1 --gt-attn-dropout 0.2 --gps-layers 2 --use-mordred-features \
                --mordred-feature-dim 11 --mordred-feature-path "$mordred_lookup" \
                --use-fifth-mechanistic-descriptors --fifth-mechanistic-descriptor-dim 12 \
                --fifth-mechanistic-descriptor-path "$fifth_lookup" --use-component-aux-features \
                --execution-max-epochs "$MAX_EPOCHS" --include-test 2>&1 | tee "$O13E_ROOT/logs/seed${seed}_${target_group}.log"
        fi
        PYTHONPATH=. "$PYTHON" scripts/diagnostics/audit_o13e_strict_scaling_config.py \
            --o13c-effective "$c_run/effective_config.yaml" --o13e-effective "$e_run/effective_config.yaml" \
            --strict-mordred "$mordred_lookup" --strict-fifth "$fifth_lookup" \
            --mordred-scaler-metadata "${mordred_lookup%.csv}.json" \
            --fifth-scaler-metadata "${fifth_lookup%.csv}.json" --output "$e_run/config_diff_audit.json"
    done
done

PYTHONPATH=. "$PYTHON" scripts/diagnostics/evaluate_o12_10seed_corresponding_splits.py \
    --model-root "$O13C_ROOT" --manifest-root "$MANIFEST_ROOT" \
    --output-dir "$O13C_ROOT/corresponding_split_single_inference" --seeds "${SPLIT_SEEDS[@]}" --target-groups "${TARGET_GROUPS[@]}"
PYTHONPATH=. "$PYTHON" scripts/diagnostics/evaluate_o12_10seed_corresponding_splits.py \
    --model-root "$O13E_ROOT" --manifest-root "$MANIFEST_ROOT" \
    --output-dir "$O13E_ROOT/corresponding_split_single_inference" --seeds "${SPLIT_SEEDS[@]}" --target-groups "${TARGET_GROUPS[@]}"

PYTHONPATH=. "$PYTHON" scripts/diagnostics/compare_o13e_strict_scaling.py \
    --input-csv "$INPUT_CSV" \
    --o13c-metrics "$O13C_ROOT/corresponding_split_single_inference/metrics_by_checkpoint_target.csv" \
    --o13c-predictions "$O13C_ROOT/corresponding_split_single_inference/predictions_by_checkpoint.csv" \
    --o13e-metrics "$O13E_ROOT/corresponding_split_single_inference/metrics_by_checkpoint_target.csv" \
    --o13e-predictions "$O13E_ROOT/corresponding_split_single_inference/predictions_by_checkpoint.csv" \
    --output-dir "$ROOT_OUT/o13e_vs_o13c_strict_scaling_comparison"

#!/usr/bin/env bash
# O13-D: frozen O12 protocol + exactly mean component-5 pooling and Fifth_class embedding.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
O12_BASELINE_ROOT="${O12_BASELINE_ROOT:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
INPUT_CSV="${INPUT_CSV:-$O12_BASELINE_ROOT/staging/20260812-sum-700_utf8.csv}"
BASE_CONFIG="${BASE_CONFIG:-$O12_BASELINE_ROOT/core4/O12_split100/source_config.yaml}"
MORDRED="${MORDRED:-results/input_graphgps_optimization/features/mordred11_train_standardized.csv}"
RANDOM_MANIFESTS="${RANDOM_MANIFESTS:-results/input_graphgps_optimization/five_split_manifests}"
OOD_MANIFESTS="${OOD_MANIFESTS:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests}"
O13B_RANDOM_ROOT="${O13B_RANDOM_ROOT:-results/input_graphgps_optimization/o13b_fifth_class_embedding_random_seed100_109}"
O13B_OOD_ROOT="${O13B_OOD_ROOT:-results/input_graphgps_optimization/o13b_fifth_class_embedding_fifth_identity_ood_seed100_109}"
O13C_RANDOM_ROOT="${O13C_RANDOM_ROOT:-results/input_graphgps_optimization/o13c_mean_graph_pooling_random_seed100_109}"
O13C_OOD_ROOT="${O13C_OOD_ROOT:-results/input_graphgps_optimization/o13c_mean_graph_pooling_fifth_identity_ood_seed100_109}"
RANDOM_RUNS_ROOT="${RANDOM_RUNS_ROOT:-results/input_graphgps_optimization/o13d_mean_pooling_fifth_class_random_seed100_109}"
OOD_RUNS_ROOT="${OOD_RUNS_ROOT:-results/input_graphgps_optimization/o13d_mean_pooling_fifth_class_fifth_identity_ood_seed100_109}"
AUDIT_ROOT="${AUDIT_ROOT:-results/input_graphgps_optimization/o13d_mean_pooling_fifth_class_audit}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
SPLIT_SEEDS=(100 101 102 103 104 105 106 107 108 109)
TARGET_GROUPS=(core4 norm2)

for path in "$INPUT_CSV" "$BASE_CONFIG" "$MORDRED"; do
    [[ -f "$path" ]] || { echo "Missing locked O12 file: $path" >&2; exit 2; }
done
for path in "$RANDOM_MANIFESTS/split_manifest_seed100.csv" \
            "$OOD_MANIFESTS/fifth_identity_manifest_seed100.csv" \
            "$OOD_MANIFESTS/protocol.json"; do
    [[ -f "$path" ]] || { echo "Missing frozen manifest/protocol: $path" >&2; exit 2; }
done
for target_group in "${TARGET_GROUPS[@]}"; do
    baseline_effective="$O12_BASELINE_ROOT/$target_group/O12_split100/effective_config.yaml"
    [[ -f "$baseline_effective" ]] || { echo "Missing O12 effective config: $baseline_effective" >&2; exit 2; }
    "$PYTHON" scripts/diagnostics/audit_o13d_mean_pooling_fifth_class_config.py \
        --baseline-effective "$baseline_effective" --locked-input-csv "$INPUT_CSV" \
        --locked-mordred "$MORDRED" --output "$AUDIT_ROOT/preflight_${target_group}.json"
done

# Record the deterministic global class mapping before any split is applied.
"$PYTHON" scripts/diagnostics/audit_o13b_fifth_class_encoding.py \
    --input-csv "$INPUT_CSV" --random-manifest-root "$RANDOM_MANIFESTS" \
    --ood-manifest-root "$OOD_MANIFESTS" --seeds "${SPLIT_SEEDS[@]}" \
    --output-dir "$AUDIT_ROOT/fifth_class_encoding"

run_protocol() {
    local protocol="$1" manifests="$2" runs_root="$3" manifest_prefix="$4"
    mkdir -p "$runs_root/logs"
    for target_group in "${TARGET_GROUPS[@]}"; do
        local output_activation="identity" candidate_prefix="O13D${protocol}Norm2"
        [[ "$target_group" == "core4" ]] && output_activation="sigmoid" && candidate_prefix="O13D${protocol}Core4"
        local baseline_effective="$O12_BASELINE_ROOT/$target_group/O12_split100/effective_config.yaml"
        for split_seed in "${SPLIT_SEEDS[@]}"; do
            local manifest="$manifests/${manifest_prefix}${split_seed}.csv"
            local run_dir="$runs_root/$target_group/O12_split${split_seed}"
            [[ -f "$manifest" ]] || { echo "Missing manifest: $manifest" >&2; exit 2; }
            if [[ -e "$run_dir" ]]; then
                if [[ -f "$run_dir/summary.json" && -f "$run_dir/predictions.csv" && \
                      -f "$run_dir/checkpoints/selected_best.pt" && -f "$run_dir/config_diff_audit.json" ]]; then
                    echo "Skipping completed O13-D ${protocol} ${target_group} split ${split_seed}"
                    continue
                fi
                echo "Refusing incomplete O13-D directory: $run_dir" >&2
                echo "Use fresh RANDOM_RUNS_ROOT/OOD_RUNS_ROOT; never reuse an old processed cache." >&2
                exit 1
            fi
            echo "Training O13-D ${protocol} ${target_group} split ${split_seed}"
            "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
                --config "$BASE_CONFIG" --run-dir "$run_dir" --input-csv "$INPUT_CSV" \
                --target-set "$target_group" --split-manifest "$manifest" \
                --fold "${protocol}_split${split_seed}" --group B --candidate "${candidate_prefix}${split_seed}" \
                --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
                --graph-pooling mean --use-fifth-class-embedding --output-activation "$output_activation" \
                --seed 43 --base-lr 0.001 --weight-decay 1e-5 --batch-size 8 \
                --warmup-epochs 50 --early-stop-patience "$EARLY_STOP_PATIENCE" \
                --gt-dropout 0.1 --gt-attn-dropout 0.2 --gps-layers 2 \
                --use-mordred-features --mordred-feature-dim 11 --mordred-feature-path "$MORDRED" \
                --use-component-aux-features --execution-max-epochs "$MAX_EPOCHS" --include-test \
                2>&1 | tee "$runs_root/logs/O13D_${protocol}_${target_group}_split${split_seed}.log"
            "$PYTHON" scripts/diagnostics/audit_o13d_mean_pooling_fifth_class_config.py \
                --baseline-effective "$baseline_effective" --candidate-effective "$run_dir/effective_config.yaml" \
                --locked-input-csv "$INPUT_CSV" --locked-mordred "$MORDRED" \
                --output "$run_dir/config_diff_audit.json"
        done
    done
    "$PYTHON" scripts/diagnostics/evaluate_o12_10seed_corresponding_splits.py \
        --model-root "$runs_root" --manifest-root "$manifests" \
        --output-dir "$runs_root/corresponding_split_single_inference" \
        --seeds "${SPLIT_SEEDS[@]}" --target-groups "${TARGET_GROUPS[@]}"
}

# Intentionally absent: fifth-only fusion, fifth identity embedding, fifth-ratio
# modulation, ratio polynomial features, new descriptors, size features, or any
# fusion/pooling modification beyond component-5 add -> mean.
run_protocol random "$RANDOM_MANIFESTS" "$RANDOM_RUNS_ROOT" "split_manifest_seed"
run_protocol fifth_identity_ood "$OOD_MANIFESTS" "$OOD_RUNS_ROOT" "fifth_identity_manifest_seed"

"$PYTHON" scripts/diagnostics/compare_o13d_mean_pooling_fifth_class.py \
    --input-csv "$INPUT_CSV" \
    --o12-random-metrics "$O12_BASELINE_ROOT/corresponding_split_single_inference/metrics_by_checkpoint_target.csv" \
    --o12-random-predictions "$O12_BASELINE_ROOT/corresponding_split_single_inference/predictions_by_checkpoint.csv" \
    --o12-ood-metrics "results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/corresponding_split_single_inference/metrics_by_checkpoint_target.csv" \
    --o12-ood-predictions "results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/corresponding_split_single_inference/predictions_by_checkpoint.csv" \
    --o13b-random-metrics "$O13B_RANDOM_ROOT/corresponding_split_single_inference/metrics_by_checkpoint_target.csv" \
    --o13b-random-predictions "$O13B_RANDOM_ROOT/corresponding_split_single_inference/predictions_by_checkpoint.csv" \
    --o13b-ood-metrics "$O13B_OOD_ROOT/corresponding_split_single_inference/metrics_by_checkpoint_target.csv" \
    --o13b-ood-predictions "$O13B_OOD_ROOT/corresponding_split_single_inference/predictions_by_checkpoint.csv" \
    --o13c-random-metrics "$O13C_RANDOM_ROOT/corresponding_split_single_inference/metrics_by_checkpoint_target.csv" \
    --o13c-random-predictions "$O13C_RANDOM_ROOT/corresponding_split_single_inference/predictions_by_checkpoint.csv" \
    --o13c-ood-metrics "$O13C_OOD_ROOT/corresponding_split_single_inference/metrics_by_checkpoint_target.csv" \
    --o13c-ood-predictions "$O13C_OOD_ROOT/corresponding_split_single_inference/predictions_by_checkpoint.csv" \
    --o13d-random-metrics "$RANDOM_RUNS_ROOT/corresponding_split_single_inference/metrics_by_checkpoint_target.csv" \
    --o13d-random-predictions "$RANDOM_RUNS_ROOT/corresponding_split_single_inference/predictions_by_checkpoint.csv" \
    --o13d-ood-metrics "$OOD_RUNS_ROOT/corresponding_split_single_inference/metrics_by_checkpoint_target.csv" \
    --o13d-ood-predictions "$OOD_RUNS_ROOT/corresponding_split_single_inference/predictions_by_checkpoint.csv" \
    --output-dir results/input_graphgps_optimization/o13d_mean_pooling_fifth_class_comparison

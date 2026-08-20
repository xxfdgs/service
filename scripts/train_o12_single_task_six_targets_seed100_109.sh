#!/usr/bin/env bash
# Train true one-output O12 models for all six properties on split seeds 100-109.
#
# Each target/seed pair starts from scratch with O12's fixed architecture and
# hyperparameters.  The dataset membership is read from the already-saved
# manifests, not regenerated.  Completed runs are skipped and interrupted
# runs resume safely.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
MANIFESTS="${MANIFESTS:-results/input_graphgps_optimization/five_split_manifests}"
RUNS_ROOT="${RUNS_ROOT:-results/input_graphgps_optimization/single_task_o12_six_targets}"
BASE_CONFIG="${BASE_CONFIG:-results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/source_config.yaml}"
MORDRED="${MORDRED:-results/input_graphgps_optimization/features/mordred11_train_standardized.csv}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
TARGETS=(EE_before EE_after Aerosolization_Efficiency mRNA_Recovery_Efficiency Norm_before Norm_after)
SPLIT_SEEDS=(100 101 102 103 104 105 106 107 108 109)

mkdir -p "$RUNS_ROOT/logs"

for target in "${TARGETS[@]}"; do
    for split_seed in "${SPLIT_SEEDS[@]}"; do
        manifest="$MANIFESTS/split_manifest_seed${split_seed}.csv"
        run_dir="$RUNS_ROOT/O12_${target}_split${split_seed}"
        candidate="O12Single_${target}_S${split_seed}"
        if [[ ! -f "$manifest" ]]; then
            echo "Missing fixed split manifest: $manifest" >&2
            exit 1
        fi
        if [[ -f "$run_dir/summary.json" && -f "$run_dir/predictions.csv" ]]; then
            echo "Skipping completed ${target}, split ${split_seed}"
            continue
        fi
        resume_args=()
        if [[ -f "$run_dir/resume_state.pt" ]]; then
            resume_args+=(--resume)
        elif [[ -d "$run_dir" ]]; then
            resume_args+=(--restart-incomplete)
        fi
        echo "Training O12 single-task ${target}, split ${split_seed}"
        "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
            --config "$BASE_CONFIG" \
            --run-dir "$run_dir" \
            --single-target "$target" \
            --split-manifest "$manifest" \
            --fold "split${split_seed}" --group B --candidate "$candidate" \
            --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
            --seed 43 --base-lr 0.001 --weight-decay 1e-5 \
            --gt-dropout 0.1 --gt-attn-dropout 0.2 \
            --use-mordred-features --mordred-feature-dim 11 \
            --mordred-feature-path "$MORDRED" \
            --use-component-aux-features \
            --execution-max-epochs "$MAX_EPOCHS" --include-test \
            "${resume_args[@]}" \
            > "$RUNS_ROOT/logs/O12_${target}_split${split_seed}.log" 2>&1
    done
done

"$PYTHON" scripts/diagnostics/summarize_selected_checkpoints_test.py \
    --runs-root "$RUNS_ROOT" \
    --output-dir "$RUNS_ROOT/checkpoint_test_metrics"

#!/usr/bin/env bash
# Train O12 and O22 on Norm_before / Norm_after for every saved input split.
#
# Each run uses the same 36 split manifests already used for the four core
# properties.  Completed runs are skipped; interrupted runs resume from their
# saved state, so rerunning this file is safe.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
MANIFESTS="${MANIFESTS:-results/input_graphgps_optimization/five_split_manifests}"
RUNS_ROOT="${RUNS_ROOT:-results/input_graphgps_optimization/norm2_five_split_runs}"
BASE_CONFIG="${BASE_CONFIG:-results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/source_config.yaml}"
MORDRED="${MORDRED:-results/input_graphgps_optimization/features/mordred11_train_standardized.csv}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"

mkdir -p "$RUNS_ROOT/logs"

run_model() {
    local model="$1"
    local fusion_type="$2"
    local manifest split_seed run_dir candidate

    while IFS= read -r manifest; do
        split_seed="${manifest##*/split_manifest_seed}"
        split_seed="${split_seed%.csv}"
        run_dir="$RUNS_ROOT/${model}_split${split_seed}"
        candidate="${model}N${split_seed}"

        if [[ -f "$run_dir/summary.json" && -f "$run_dir/predictions.csv" ]]; then
            echo "Skipping completed ${model} split ${split_seed}"
            continue
        fi

        local resume_args=()
        if [[ -f "$run_dir/resume_state.pt" ]]; then
            resume_args+=(--resume)
        elif [[ -d "$run_dir" ]]; then
            resume_args+=(--restart-incomplete)
        fi

        echo "Training ${model} split ${split_seed}"
        "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
            --config "$BASE_CONFIG" \
            --run-dir "$run_dir" \
            --target-set norm2 \
            --split-manifest "$manifest" \
            --fold "split${split_seed}" --group B --candidate "$candidate" \
            --fusion-type "$fusion_type" --head-type baseline --model-type OneHotEmbedGPS \
            --seed 43 --base-lr 0.001 --weight-decay 1e-5 \
            --gt-dropout 0.1 --gt-attn-dropout 0.2 \
            --use-mordred-features --mordred-feature-dim 11 \
            --mordred-feature-path "$MORDRED" \
            --use-component-aux-features \
            --execution-max-epochs "$MAX_EPOCHS" --include-test \
            "${resume_args[@]}" \
            > "$RUNS_ROOT/logs/${model}_split${split_seed}.log" 2>&1
    done < <(find "$MANIFESTS" -maxdepth 1 -type f -name 'split_manifest_seed*.csv' | sort)
}

run_model O12 concat_mlp
run_model O22 gated_concat

# Exports a row for every selected checkpoint and each corresponding test set,
# plus target-level and macro averages across all completed saved splits.
"$PYTHON" scripts/diagnostics/summarize_selected_checkpoints_test.py \
    --runs-root "$RUNS_ROOT" \
    --output-dir "$RUNS_ROOT/checkpoint_test_metrics"

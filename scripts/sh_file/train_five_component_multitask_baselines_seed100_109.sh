#!/usr/bin/env bash
# Queue GCN/GIN/MPNN/Transformer/MLP multi-task baselines on fixed splits.
#SBATCH -o %j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH -p gpu4090
#SBATCH --gres=gpu:2

module load apps/anaconda3/2021.05
module load compiler/cuda/12.4

source /public/software/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate biology-prediction_gpu

INPUT_CSV="${INPUT_CSV:-results/new_dataset_benchmark_20260713/input_sanitized_utf8.csv}"
MANIFESTS="${MANIFESTS:-results/input_graphgps_optimization/five_split_manifests}"
RUNS_ROOT="${RUNS_ROOT:-results/input_graphgps_optimization/multitask_baselines_seed100_109}"
EPOCHS="${EPOCHS:-300}"
MODELS=(GCN GIN MPNN Transformer MLP)
TARGET_GROUPS=(core4 norm2)
SPLIT_SEEDS=(100 101 102 103 104 105 106 107 108 109)


mkdir -p "$RUNS_ROOT/logs"
for model in "${MODELS[@]}"; do
    for target_group in "${TARGET_GROUPS[@]}"; do
        for split_seed in "${SPLIT_SEEDS[@]}"; do
            manifest="$MANIFESTS/split_manifest_seed${split_seed}.csv"
            run_dir="$RUNS_ROOT/${model}_${target_group}_split${split_seed}"
            if [[ ! -f "$manifest" ]]; then
                echo "Missing fixed split manifest: $manifest" >&2
                exit 1
            fi
            if [[ -f "$run_dir/summary.json" && -f "$run_dir/predictions.csv" ]]; then
                echo "Skipping completed ${model}/${target_group}/split${split_seed}"
                continue
            fi
            resume_args=()
            if [[ -f "$run_dir/resume_state.pt" ]]; then
                resume_args+=(--resume)
            fi
            echo "Training ${model}/${target_group}/split${split_seed}"
            python scripts/diagnostics/run_five_component_multitask_baseline.py \
                --input-csv "$INPUT_CSV" --split-manifest "$manifest" --run-dir "$run_dir" \
                --model "$model" --target-group "$target_group" --seed 43 \
                --epochs "$EPOCHS" --batch-size 8 --hidden-dim 64 --layers 2 --dropout .1 \
                --lr .001 --weight-decay 1e-5 --warmup-epochs 50 \
                --early-stop-patience 50 --early-stop-min-delta .001 \
                "${resume_args[@]}" > "$RUNS_ROOT/logs/${model}_${target_group}_split${split_seed}.log" 2>&1
        done
    done
done

"$PYTHON" scripts/diagnostics/summarize_five_component_multitask_baselines.py \
    --runs-root "$RUNS_ROOT" --output-dir "$RUNS_ROOT/test_metrics"

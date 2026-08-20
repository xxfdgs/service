#!/usr/bin/env bash
# Queue GCN/GIN/MPNN/Transformer/MLP multi-task baselines on fixed splits.
#SBATCH -o %j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -N 1
#SBATCH -p gpu4090
#SBATCH --gres=gpu:1

module load apps/anaconda3/2021.05
module load compiler/cuda/12.4

source /public/software/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate biology-prediction_gpu

echo "SCRIPT = $(realpath "$0")"
echo "PWD = $(pwd)"
echo "Looking for = $(realpath -m "$SOURCE_INPUT_CSV")"


# Reproduce experiment I: O12 multitask core4 + norm2 models trained only on
# the original 700-row input data, using the fixed 100--109 split manifests.
#
# IMPORTANT: this script is for a loader/ratio fix.  Every new run must build
# a fresh isolated PyG processed cache.  Do not pass --reuse-existing-cache,
# and do not resume a directory created before the loader fix: its saved
# train_5.pt / val_5.pt / test_5.pt may contain stale ratio values.
#
# The defaults below intentionally mirror the saved experiment-I run settings:
#   OneHotEmbedGPS, 2 GPS layers (64 hidden dimensions from source_config),
#   concat_mlp fusion, baseline head, 11 Mordred descriptors, component
#   auxiliary features, MAE loss, LR=1e-3, batch size 8, and warm-up 50.
#
# Core4 uses a sigmoid readout in normalized [0, 1] space, so reported
# efficiency predictions are physically bounded to [0, 100].  Norm2 retains
# its unbounded identity readout because it is not a percentage target.
# Results go to a new directory by default.  Set RUNS_ROOT explicitly only if
# you deliberately want to use a different output location.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON="$(which python)"
BASE="${BASE:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
INPUT="$BASE/staging/20260812-sum-700_utf8.csv"; CONFIG="$BASE/core4/O12_split100/source_config.yaml"; MANIFESTS="results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests"; OUT="${OUT:-results/input_graphgps_optimization/o13g_structured_norm2}"
for seed in {100..109}; do
 m="$MANIFESTS/fifth_identity_manifest_seed$seed.csv"; pre="$OUT/preprocessing/seed$seed"; mkdir -p "$pre" "$OUT/logs"
 PYTHONPATH=. "$PYTHON" scripts/diagnostics/build_o13g_structured_features.py --input-csv "$INPUT" --manifest "$m" --raw-output "$OUT/o13g_fifth_structured_features_raw.csv" --lookup-output "$pre/structured.csv" --audit "$pre/feature_audit.json"
 run="$OUT/norm2/O13G_norm2_split$seed"; [[ ! -e "$run" ]] || { echo "existing $run" >&2; exit 1; }
 PYTHONPATH=. "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py --config "$CONFIG" --run-dir "$run" --input-csv "$INPUT" --target-set norm2 --split-manifest "$m" --fold "fifth_identity_ood_split$seed" --group B --candidate "O13G_norm2_split$seed" --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS --graph-pooling mean --seed 43 --base-lr 0.001 --weight-decay 1e-5 --batch-size 8 --warmup-epochs 10 --early-stop-patience 50 --gt-dropout 0.1 --gt-attn-dropout 0.2 --gps-layers 2 --disable-mordred-features --use-component-aux-features --use-fifth-class-embedding --use-fifth-structured-features --fifth-structured-feature-path "$pre/structured.csv" --execution-max-epochs 300 --include-test 2>&1 | tee "$OUT/logs/seed$seed.log"
done





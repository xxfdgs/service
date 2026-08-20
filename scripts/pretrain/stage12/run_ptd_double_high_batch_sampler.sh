#!/usr/bin/env bash
# Strict internal-only Experiment E: train-only batch coverage for double >1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
SOURCE="results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline/staging/20260812-sum-700_utf8.csv"
CONFIG="results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline/core4/O12_split100/source_config.yaml"
MANIFEST_ROOT="results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests"
REFERENCE="results/fifth_pretraining/stage6_strict_no_mordred_fifth_ood_baseline/P1_PT_D_strict_no_mordred"
PTD="results/fifth_pretraining/stage4_graphgps_pretraining/PT_D/checkpoints/best_comp5_encoder_state_dict.pt"
OUT_ROOT="${OUT_ROOT:-results/fifth_pretraining/stage12_ptd_double_gt1_batch_sampler/batch_sampler_screen}"
read -r -a SPLIT_SEEDS <<< "${SPLIT_SEEDS:-100 101 102}"
mkdir -p "$OUT_ROOT/S1/logs"

for split_seed in "${SPLIT_SEEDS[@]}"; do
    run_dir="$OUT_ROOT/S1/split${split_seed}"
    if [[ -f "$run_dir/selected_best.pt" ]]; then
        echo "[stage12] completed: $run_dir"
        continue
    fi
    [[ ! -e "$run_dir" ]] || { echo "Refusing to overwrite incomplete run: $run_dir" >&2; exit 2; }
    "$PYTHON_BIN" -u scripts/diagnostics/run_fusion_head_experiment.py \
        --config "$CONFIG" \
        --run-dir "$run_dir" \
        --input-csv "$SOURCE" \
        --component-vocab-source "$SOURCE" \
        --target-set norm2 --single-target Norm_before \
        --split-manifest "$MANIFEST_ROOT/fifth_identity_manifest_seed${split_seed}.csv" \
        --fold "fifth_identity_ood_split${split_seed}" \
        --group B --candidate Stage12_S1_PTD_BatchSampler \
        --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
        --graph-pooling mean --use-fifth-class-embedding --output-activation identity \
        --seed 43 --base-lr 0.001 --weight-decay 1e-5 --batch-size 8 \
        --warmup-epochs 50 --early-stop-patience 50 --gt-dropout 0.1 --gt-attn-dropout 0.2 \
        --gps-layers 2 --disable-mordred-features --use-component-aux-features \
        --comp5-pretrained-checkpoint "$PTD" --comp5-pretrain-label P1_PT_D_strict_no_mordred \
        --training-loss huber --huber-beta 0.1 --norm-threshold-report-only \
        --min-double-high-per-batch 1 \
        --execution-max-epochs 300 --include-test --require-membership-count 700 \
        --membership-reference-run-dir "$REFERENCE/split${split_seed}" --require-fresh-cache \
        2>&1 | tee "$OUT_ROOT/S1/logs/split${split_seed}.log"
done

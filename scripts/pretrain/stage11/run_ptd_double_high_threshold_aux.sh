#!/usr/bin/env bash
# Stage 11: strict PT-D auxiliary y>1 head screen (Experiment D).
#
# This is a representation regularizer only. The continuous regression head is
# always the reported prediction; the classifier is never used to overwrite or
# threshold-shift it. The explicit crossing penalty is set to zero so the sole
# intervention is lambda * BCE(shared_fusion_logit, Norm_before>1).
set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
RUNNER="${RUNNER:-scripts/diagnostics/run_fusion_head_experiment.py}"
BASE="${BASE:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
INPUT="${INPUT:-$BASE/staging/20260812-sum-700_utf8.csv}"
CONFIG="${CONFIG:-$BASE/core4/O12_split100/source_config.yaml}"
MANIFESTS="${MANIFESTS:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests}"
PT_D="${PT_D:-results/fifth_pretraining/stage4_graphgps_pretraining/PT_D/checkpoints/best_comp5_encoder_state_dict.pt}"
P1_REFERENCE="${P1_REFERENCE:-results/fifth_pretraining/stage6_strict_no_mordred_fifth_ood_baseline/P1_PT_D_strict_no_mordred}"
OUT="${OUT:-results/fifth_pretraining/stage11_ptd_double_gt1_threshold_aux/threshold_aux_screen}"
read -r -a SPLIT_SEEDS <<< "${SPLIT_SEEDS:-100 101 102}"
read -r -a VARIANTS <<< "${VARIANTS:-C005 C010 C020}"

[[ -x "$PYTHON" && -f "$RUNNER" && -f "$INPUT" && -f "$CONFIG" && -f "$PT_D" ]] || {
    echo "Missing Stage 11 strict PT-D prerequisite" >&2; exit 2;
}

lambda_for() {
    case "$1" in
        C005) printf '0.05' ;;
        C010) printf '0.10' ;;
        C020) printf '0.20' ;;
        *) echo "Unknown VARIANTS entry: $1" >&2; exit 2 ;;
    esac
}

for seed in "${SPLIT_SEEDS[@]}"; do
    manifest="$MANIFESTS/fifth_identity_manifest_seed${seed}.csv"
    reference="$P1_REFERENCE/split${seed}"
    [[ -f "$manifest" && -f "$reference/predictions.csv" ]] || {
        echo "Missing frozen manifest or P1 reference for split $seed" >&2; exit 2;
    }
    for variant in "${VARIANTS[@]}"; do
        lambda="$(lambda_for "$variant")"
        run="$OUT/$variant/split${seed}"
        [[ ! -e "$run" ]] || { echo "Refusing to overwrite $run" >&2; exit 2; }
        mkdir -p "$OUT/$variant/logs"
        PYTHONPATH=. "$PYTHON" -u "$RUNNER" \
            --config "$CONFIG" --run-dir "$run" --input-csv "$INPUT" \
            --component-vocab-source "$INPUT" --target-set norm2 --single-target Norm_before \
            --split-manifest "$manifest" --fold "fifth_identity_ood_split${seed}" \
            --group B --candidate "Stage11_${variant}_PTDThresholdAux" \
            --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
            --graph-pooling mean --use-fifth-class-embedding --output-activation identity \
            --seed 43 --base-lr 0.001 --weight-decay 1e-5 --batch-size 8 \
            --warmup-epochs 50 --early-stop-patience 50 --gt-dropout 0.1 --gt-attn-dropout 0.2 \
            --gps-layers 2 --disable-mordred-features --use-component-aux-features \
            --comp5-pretrained-checkpoint "$PT_D" --comp5-pretrain-label "P1_PT_D_strict_no_mordred" \
            --training-loss huber --huber-beta 0.1 --enable-norm-threshold-aware \
            --norm-threshold 1.0 --norm-cls-loss-weight "$lambda" --norm-fn-loss-weight 0.0 \
            --norm-positive-reg-weight 1.0 --norm-underprediction-weight 0.0 \
            --execution-max-epochs 300 --include-test --require-membership-count 700 \
            --membership-reference-run-dir "$reference" --require-fresh-cache \
            2>&1 | tee "$OUT/$variant/logs/split${seed}.log"
    done
done

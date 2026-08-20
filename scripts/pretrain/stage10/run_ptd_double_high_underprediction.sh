#!/usr/bin/env bash
# Stage 10: strict PT-D asymmetric continuous underprediction-loss screen.
#
# The only objective change from the strict P1 PT-D No-Mordred baseline is
# gamma * mean(relu(Norm_before - prediction)^2) on train rows satisfying
# Fifth_class=double and Norm_before>1.  It deliberately does not stack the
# Stage-9 positive-region weighting, so this is an interpretable Experiment-B
# ablation.  All split membership and checkpoint selection remain frozen.
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
OUT="${OUT:-results/fifth_pretraining/stage10_ptd_double_gt1_underprediction/underprediction_screen_norm_after}"
read -r -a SPLIT_SEEDS <<< "${SPLIT_SEEDS:-103 104 105 106 107 108 109}"
read -r -a VARIANTS <<< "${VARIANTS:-U025 U050 U100}"

[[ -x "$PYTHON" && -f "$RUNNER" && -f "$INPUT" && -f "$CONFIG" && -f "$PT_D" ]] || {
    echo "Missing Stage 10 strict PT-D prerequisite" >&2; exit 2;
}

gamma_for() {
    case "$1" in
        U025) printf '0.25' ;;
        U050) printf '0.50' ;;
        U100) printf '1.00' ;;
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
        gamma="$(gamma_for "$variant")"
        run="$OUT/$variant/split${seed}"
        [[ ! -e "$run" ]] || { echo "Refusing to overwrite $run" >&2; exit 2; }
        mkdir -p "$OUT/$variant/logs"
        PYTHONPATH=. "$PYTHON" -u "$RUNNER" \
            --config "$CONFIG" --run-dir "$run" --input-csv "$INPUT" \
            --component-vocab-source "$INPUT" --target-set norm2 --single-target Norm_after \
            --split-manifest "$manifest" --fold "fifth_identity_ood_split${seed}" \
            --group B --candidate "Stage10_${variant}_PTDUnderprediction" \
            --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
            --graph-pooling mean --use-fifth-class-embedding --output-activation identity \
            --seed 43 --base-lr 0.001 --weight-decay 1e-5 --batch-size 8 \
            --warmup-epochs 50 --early-stop-patience 50 --gt-dropout 0.1 --gt-attn-dropout 0.2 \
            --gps-layers 2 --disable-mordred-features --use-component-aux-features \
            --comp5-pretrained-checkpoint "$PT_D" --comp5-pretrain-label "P1_PT_D_strict_no_mordred" \
            --training-loss huber --huber-beta 0.1 --norm-threshold-report-only \
            --norm-threshold 1.0 --norm-positive-reg-weight 1.0 \
            --norm-underprediction-weight "$gamma" --execution-max-epochs 300 --include-test \
            --require-membership-count 700 --membership-reference-run-dir "$reference" \
            --require-fresh-cache \
            2>&1 | tee "$OUT/$variant/logs/split${seed}.log"
    done
done

#!/usr/bin/env bash
# Stage 9: compact, strict PT-D objective screen for double Norm_before > 1.
#
# This intentionally preserves every P1_PT_D strict No-Mordred input/backbone
# setting.  The only changed training variable is the Huber objective plus the
# explicit train-only weight for double true-high labels.  The test split is
# never read until the validation-selected checkpoint is frozen.
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
OUT="${OUT:-results/fifth_pretraining/stage9_ptd_double_gt1_optimization/weighted_huber_screen}"
read -r -a SPLIT_SEEDS <<< "${SPLIT_SEEDS:-100 101 102}"
# H0 distinguishes the Huber loss itself from the requested A1.5/A2/A3
# positive-region weighting.  This is a deliberately four-variant screen,
# rather than an open-ended hyperparameter search.
read -r -a VARIANTS <<< "${VARIANTS:-H0 H15 H20 H30}"

[[ -x "$PYTHON" && -f "$RUNNER" && -f "$INPUT" && -f "$CONFIG" && -f "$PT_D" ]] || {
    echo "Missing Stage 9 strict PT-D prerequisite" >&2; exit 2;
}

weight_for() {
    case "$1" in
        H0) printf '1.0' ;; H15) printf '1.5' ;; H20) printf '2.0' ;; H30) printf '3.0' ;;
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
        weight="$(weight_for "$variant")"
        run="$OUT/$variant/split${seed}"
        [[ ! -e "$run" ]] || { echo "Refusing to overwrite $run" >&2; exit 2; }
        mkdir -p "$OUT/$variant/logs"
        PYTHONPATH=. "$PYTHON" -u "$RUNNER" \
            --config "$CONFIG" --run-dir "$run" --input-csv "$INPUT" \
            --component-vocab-source "$INPUT" --target-set norm2 --single-target Norm_before \
            --split-manifest "$manifest" --fold "fifth_identity_ood_split${seed}" \
            --group B --candidate "Stage9_${variant}_PTDWeightedHuber" \
            --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
            --graph-pooling mean --use-fifth-class-embedding --output-activation identity \
            --seed 43 --base-lr 0.001 --weight-decay 1e-5 --batch-size 8 \
            --warmup-epochs 50 --early-stop-patience 50 --gt-dropout 0.1 --gt-attn-dropout 0.2 \
            --gps-layers 2 --disable-mordred-features --use-component-aux-features \
            --comp5-pretrained-checkpoint "$PT_D" --comp5-pretrain-label "P1_PT_D_strict_no_mordred" \
            --training-loss huber --huber-beta 0.1 --norm-threshold-report-only \
            --norm-threshold 1.0 --norm-positive-reg-weight "$weight" \
            --norm-underprediction-weight 0.0 --execution-max-epochs 300 --include-test \
            --require-membership-count 700 --membership-reference-run-dir "$reference" \
            --require-fresh-cache \
            2>&1 | tee "$OUT/$variant/logs/split${seed}.log"
    done
done

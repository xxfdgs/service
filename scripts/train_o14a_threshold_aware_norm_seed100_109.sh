#!/usr/bin/env bash
# O14-A controlled objective ablation: all runs retain the O13G input/backbone
# configuration.  A0 is regression-only; A1-A3 only add the O14-A head/loss.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
BASE="${BASE:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
INPUT="${INPUT:-$BASE/staging/20260812-sum-700_utf8.csv}"
CONFIG="${CONFIG:-$BASE/core4/O12_split100/source_config.yaml}"
SPLIT_PROTOCOL="${SPLIT_PROTOCOL:-fifth_identity_ood}"

case "$SPLIT_PROTOCOL" in
    fifth_identity_ood)
        MANIFESTS="${MANIFESTS:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests}"
        manifest_name() { printf 'fifth_identity_manifest_seed%s.csv' "$1"; }
        fold_prefix="fifth_identity_ood_split"
        ;;
    random)
        MANIFESTS="${MANIFESTS:-results/input_graphgps_optimization/five_split_manifests}"
        manifest_name() { printf 'split_manifest_seed%s.csv' "$1"; }
        fold_prefix="random_split"
        ;;
    *) echo "Unsupported SPLIT_PROTOCOL: $SPLIT_PROTOCOL" >&2; exit 2 ;;
esac

OUT="${OUT:-results/input_graphgps_optimization/o14a_threshold_aware_${SPLIT_PROTOCOL}}"
read -r -a SPLIT_SEEDS <<< "${SPLIT_SEEDS:-100 101 102 103 104 105 106 107 108 109}"
read -r -a TARGETS <<< "${TARGETS:-Norm_before Norm_after}"
read -r -a ABLATIONS <<< "${ABLATIONS:-A0 A1 A2 A3}"

[[ -f "$INPUT" && -f "$CONFIG" ]] || { echo "Missing O13G baseline input/config." >&2; exit 2; }

for target in "${TARGETS[@]}"; do
    case "$target" in
        Norm_before|Norm_after) ;;
        *) echo "O14-A only supports Norm_before or Norm_after, not $target" >&2; exit 2 ;;
    esac
    target_slug="$(printf '%s' "$target" | tr '[:upper:]' '[:lower:]')"
    for ablation in "${ABLATIONS[@]}"; do
        case "$ablation" in
            A0)
                # Regression-only control: no O14-A head; baseline objective exactly.
                objective_args=(--norm-threshold-report-only --norm-threshold 1.0
                    --norm-cls-loss-weight 0.0 --norm-fn-loss-weight 0.0
                    --norm-positive-reg-weight 1.0)
                label="BaselineRegression"
                ;;
            A1)
                objective_args=(--enable-norm-threshold-aware --norm-threshold 1.0
                    --norm-cls-loss-weight 0.5 --norm-fn-loss-weight 0.0
                    --norm-positive-reg-weight 1.0)
                label="AuxClassification"
                ;;
            A2)
                objective_args=(--enable-norm-threshold-aware --norm-threshold 1.0
                    --norm-cls-loss-weight 0.5 --norm-fn-loss-weight 1.0
                    --norm-positive-reg-weight 1.0)
                label="ClassificationCrossing"
                ;;
            A3)
                objective_args=(--enable-norm-threshold-aware --norm-threshold 1.0
                    --norm-cls-loss-weight 0.5 --norm-fn-loss-weight 1.0
                    --norm-positive-reg-weight 1.5)
                label="CrossingPositiveWeight15"
                ;;
            *) echo "Unsupported ABLATIONS value: $ablation" >&2; exit 2 ;;
        esac
        for split_seed in "${SPLIT_SEEDS[@]}"; do
            manifest="$MANIFESTS/$(manifest_name "$split_seed")"
            [[ -f "$manifest" ]] || { echo "Missing frozen manifest: $manifest" >&2; exit 2; }
            preprocessing="$OUT/preprocessing/$ablation/$target_slug/seed$split_seed"
            run="$OUT/$ablation/$target_slug/O14A_${ablation}_${target_slug}_split$split_seed"
            mkdir -p "$preprocessing" "$OUT/logs/$ablation/$target_slug"
            [[ ! -e "$run" ]] || { echo "Existing run (refusing overwrite): $run" >&2; exit 1; }
            PYTHONPATH=. "$PYTHON" scripts/diagnostics/build_o13g_structured_features.py \
                --input-csv "$INPUT" --manifest "$manifest" \
                --raw-output "$OUT/o13g_fifth_structured_features_raw.csv" \
                --lookup-output "$preprocessing/structured.csv" \
                --audit "$preprocessing/feature_audit.json"
            PYTHONPATH=. "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
                --config "$CONFIG" --run-dir "$run" --input-csv "$INPUT" \
                --target-set norm2 --single-target "$target" --split-manifest "$manifest" \
                --fold "${fold_prefix}$split_seed" --group B \
                --candidate "O14A_${ablation}_${target_slug}_${label}" \
                --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
                --graph-pooling mean --seed 43 --base-lr 0.001 --weight-decay 1e-5 \
                --batch-size 8 --warmup-epochs 10 --early-stop-patience 50 \
                --gt-dropout 0.1 --gt-attn-dropout 0.2 --gps-layers 2 \
                --disable-mordred-features --use-component-aux-features \
                --use-fifth-class-embedding --use-fifth-structured-features \
                --fifth-structured-feature-path "$preprocessing/structured.csv" \
                "${objective_args[@]}" --execution-max-epochs 300 --include-test \
                2>&1 | tee "$OUT/logs/$ablation/$target_slug/seed$split_seed.log"
        done
    done
done

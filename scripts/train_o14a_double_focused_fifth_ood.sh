#!/usr/bin/env bash
# O14-A primary protocol.  It intentionally accepts only frozen Fifth-identity
# OOD manifests: random splits are a later, explicitly secondary control.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
BASE="${BASE:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
FULL_INPUT="${FULL_INPUT:-$BASE/staging/20260812-sum-700_utf8.csv}"
CONFIG="${CONFIG:-$BASE/core4/O12_split100/source_config.yaml}"
FULL_MANIFESTS="${FULL_MANIFESTS:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests}"
OUT="${OUT:-results/input_graphgps_optimization/o14a_double_focused_fifth_identity_ood_sigmoid_penalty_20260812_high_20}"

# The primary protocol is deliberately not configurable to random.  The guard
# prevents a typo or inherited shell environment from silently changing it.
SPLIT_PROTOCOL="${SPLIT_PROTOCOL:-fifth_identity_ood}"
[[ "$SPLIT_PROTOCOL" == "fifth_identity_ood" ]] || {
    echo "O14-A primary runs require SPLIT_PROTOCOL=fifth_identity_ood; got $SPLIT_PROTOCOL" >&2
    exit 2
}

# Stage 1 default: matched ordinary-regression Full vs Double.  For Stage 2,
# choose DOMAIN=full or DOMAIN=double after reviewing Stage 1, then set e.g.
# ABLATIONS='A1 A2 A3'.  DOMAIN=both is intentionally rejected for A1+.
DOMAIN="${DOMAIN:-full}"
ABLATIONS="${ABLATIONS:-A0}"
read -r -a SPLIT_SEEDS <<< "${SPLIT_SEEDS:-100 101 102 103 104 105 106 107 108 109}"
read -r -a TARGETS <<< "${TARGETS:-Norm_before Norm_after}"
read -r -a ABLATION_LIST <<< "$ABLATIONS"

[[ -f "$FULL_INPUT" && -f "$CONFIG" && -d "$FULL_MANIFESTS" ]] || {
    echo "Missing O13 baseline input/config/frozen Fifth-OOD manifests." >&2; exit 2;
}
case "$DOMAIN" in full|double|both) ;; *) echo "DOMAIN must be full, double, or both" >&2; exit 2;; esac
for ablation in "${ABLATION_LIST[@]}"; do
    case "$ablation" in A0|A1|A2|A3) ;; *) echo "Unsupported ablation: $ablation" >&2; exit 2;; esac
    if [[ "$ablation" != A0 && "$DOMAIN" == both ]]; then
        echo "Run Stage 1 first: A1/A2/A3 require DOMAIN=full or DOMAIN=double." >&2; exit 2
    fi
done

mkdir -p "$OUT"
PYTHONPATH=. "$PYTHON" scripts/diagnostics/create_o14a_double_fifth_ood_protocol.py \
    --input-csv "$FULL_INPUT" --full-manifest-dir "$FULL_MANIFESTS" \
    --output-dir "$OUT/double_protocol" --seeds "${SPLIT_SEEDS[@]}"
DOUBLE_INPUT="$OUT/double_protocol/o14a_double_input.csv"

if [[ "$DOMAIN" == both ]]; then DOMAINS=(full double); else DOMAINS=("$DOMAIN"); fi

for domain in "${DOMAINS[@]}"; do
    if [[ "$domain" == full ]]; then
        INPUT="$FULL_INPUT"
        MANIFEST_DIR="$FULL_MANIFESTS"
        manifest_name() { printf 'fifth_identity_manifest_seed%s.csv' "$1"; }
        domain_title="Full"
    else
        INPUT="$DOUBLE_INPUT"
        MANIFEST_DIR="$OUT/double_protocol"
        manifest_name() { printf 'double_fifth_identity_manifest_seed%s.csv' "$1"; }
        domain_title="Double"
    fi
    for target in "${TARGETS[@]}"; do
        case "$target" in
            Norm_before) target_slug="norm_before"; target_title="NormBefore" ;;
            Norm_after) target_slug="norm_after"; target_title="NormAfter" ;;
            *) echo "O14-A supports only Norm_before/Norm_after, got $target" >&2; exit 2 ;;
        esac
        for ablation in "${ABLATION_LIST[@]}"; do
            case "$ablation" in
                A0)
                    objective_args=(--norm-threshold-report-only --norm-threshold 1.0
                        --norm-cls-loss-weight 0.0 --norm-fn-loss-weight 0.0
                        --norm-positive-reg-weight 1.0)
                    objective_title="OrdinaryRegression"
                    ;;
            esac
            for split_seed in "${SPLIT_SEEDS[@]}"; do
                manifest="$MANIFEST_DIR/$(manifest_name "$split_seed")"
                [[ -f "$manifest" ]] || { echo "Missing frozen OOD manifest: $manifest" >&2; exit 2; }
                run_name="O14${ablation}${domain_title}_FifthOOD_${target_title}_seed${split_seed}"
                run="$OUT/$ablation/$domain/$target_slug/$run_name"
                preprocessing="$OUT/preprocessing/$ablation/$domain/$target_slug/seed$split_seed"
                mkdir -p "$preprocessing" "$OUT/logs/$ablation/$domain/$target_slug"
                [[ ! -e "$run" ]] || { echo "Existing run (refusing overwrite): $run" >&2; exit 1; }
                PYTHONPATH=. "$PYTHON" scripts/diagnostics/build_o13g_structured_features.py \
                    --input-csv "$INPUT" --manifest "$manifest" \
                    --raw-output "$OUT/o13g_fifth_structured_features_raw.csv" \
                    --lookup-output "$preprocessing/structured.csv" \
                    --audit "$preprocessing/feature_audit.json"
                PYTHONPATH=. "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
                    --config "$CONFIG" --run-dir "$run" --input-csv "$INPUT" \
                    --component-vocab-source "$FULL_INPUT" --training-domain "$domain" \
                    --target-set norm2 --single-target "$target" --split-manifest "$manifest" \
                    --fold "fifth_identity_ood_${domain}_split${split_seed}" --group B \
                    --candidate "$run_name-$objective_title" \
                    --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
                    --graph-pooling mean --seed 43 --base-lr 0.001 --weight-decay 1e-5 \
                    --batch-size 8 --warmup-epochs 10 --early-stop-patience 50 \
                    --gt-dropout 0.1 --gt-attn-dropout 0.2 --gps-layers 2 \
                    --disable-mordred-features --use-component-aux-features \
                    --use-fifth-class-embedding --use-fifth-structured-features \
                    --fifth-structured-feature-path "$preprocessing/structured.csv" \
                    --fifth-aa-vocab-size 12 --fifth-terminal-vocab-size 5 \
                    "${objective_args[@]}" --norm-threshold-selection-mae-tolerance 0.05 \
                    --execution-max-epochs 300 --include-test \
                    --enable-norm-sigmoid-weighting --norm-weight-low 0.1 --norm-weight-high 20\
                    2>&1 | tee "$OUT/logs/$ablation/$domain/$target_slug/seed$split_seed.log"
            done
        done
    done
done

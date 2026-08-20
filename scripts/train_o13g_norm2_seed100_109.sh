#!/usr/bin/env bash
set -euo pipefail; cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
BASE="${BASE:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
INPUT="$BASE/staging/20260812-sum-700_utf8.csv"; CONFIG="$BASE/core4/O12_split100/source_config.yaml"

# The default preserves O13G's Fifth-identity OOD protocol.  Set
# SPLIT_PROTOCOL=random to reuse the frozen historical row-random manifests
# used by O12/O13C (five_split_manifests).  MANIFESTS remains an escape hatch
# for a separately frozen manifest directory with the same file naming.
SPLIT_PROTOCOL="${SPLIT_PROTOCOL:-fifth_identity_ood}"
case "$SPLIT_PROTOCOL" in
    fifth_identity_ood)
        default_manifests="results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests"
        manifest_name() { printf 'fifth_identity_manifest_seed%s.csv' "$1"; }
        fold_prefix="fifth_identity_ood_split"
        protocol_suffix=""
        ;;
    random)
        default_manifests="results/input_graphgps_optimization/five_split_manifests"
        manifest_name() { printf 'split_manifest_seed%s.csv' "$1"; }
        fold_prefix="random_split"
        protocol_suffix="_random"
        ;;
    *)
        echo "Unsupported SPLIT_PROTOCOL: $SPLIT_PROTOCOL (use fifth_identity_ood or random)" >&2
        exit 2
        ;;
esac
MANIFESTS="${MANIFESTS:-$default_manifests}"

# Optional class-restricted experiments.  These manifests were generated from
# train_single.csv / train_double.csv and retain their own Fifth-identity OOD
# train/val/test partition.  GROUP deliberately overrides MANIFESTS and INPUT:
# mixing a subset manifest with the 700-row CSV would misalign original_row_index.
GROUP="${GROUP:-}"
VALIDATION_CSV=""
if [[ -n "$GROUP" ]]; then
    [[ "$SPLIT_PROTOCOL" == "fifth_identity_ood" ]] || {
        echo "GROUP=$GROUP uses its frozen fifth-group manifests; do not combine it with SPLIT_PROTOCOL=$SPLIT_PROTOCOL" >&2
        exit 2
    }
    case "$GROUP" in
        single|double) ;;
        *) echo "Unsupported GROUP: $GROUP (use single or double)" >&2; exit 2 ;;
    esac
    group_root="datasets_lrx/single_double_split"
    INPUT="$group_root/train_${GROUP}.csv"
    VALIDATION_CSV="$group_root/validation_${GROUP}.csv"
    MANIFESTS="$group_root/train_${GROUP}_manifest"
    manifest_name() { printf 'random_manifest_seed%s.csv' "$1"; }
    fold_prefix="${GROUP}_fifth_identity_ood_split"
    protocol_suffix="_${GROUP}_random_split"
    [[ -f "$INPUT" && -f "$VALIDATION_CSV" ]] || {
        echo "Missing class-restricted input or validation CSV for GROUP=$GROUP" >&2; exit 2;
    }
fi

# Leave SINGLE_TARGET empty for the original two-output Norm_before/Norm_after run.
# A single target is canonicalised so that, for example, SINGLE_TARGET=norm_after
# is equivalent to SINGLE_TARGET=Norm_after.  The runner itself supports all six
# O12 targets; target_set is retained for accurate run metadata and validation.
SINGLE_TARGET="${SINGLE_TARGET:-}"
if [[ -n "$SINGLE_TARGET" ]]; then
    case "${SINGLE_TARGET,,}" in
        ee_before) SINGLE_TARGET="EE_before" ;;
        ee_after) SINGLE_TARGET="EE_after" ;;
        aerosolization_efficiency) SINGLE_TARGET="Aerosolization_Efficiency" ;;
        mrna_recovery_efficiency) SINGLE_TARGET="mRNA_Recovery_Efficiency" ;;
        norm_before) SINGLE_TARGET="Norm_before" ;;
        norm_after) SINGLE_TARGET="Norm_after" ;;
        *) echo "Unsupported SINGLE_TARGET: $SINGLE_TARGET" >&2; exit 2 ;;
    esac
    target_slug="$(printf '%s' "$SINGLE_TARGET" | tr '[:upper:]' '[:lower:]')"
    if [[ "$SINGLE_TARGET" == EE_* || "$SINGLE_TARGET" == Aerosolization_Efficiency || "$SINGLE_TARGET" == mRNA_Recovery_Efficiency ]]; then
        TARGET_SET="core4"
    else
        TARGET_SET="norm2"
    fi
    output_stem="o13g_single_${target_slug}"
    run_subdir="single_task/$target_slug"
    candidate_prefix="O13G_${target_slug}"
    single_target_args=(--single-target "$SINGLE_TARGET")
else
    TARGET_SET="norm2"
    output_stem="o13g_structured_norm2"
    run_subdir="norm2"
    candidate_prefix="O13G_norm2"
    single_target_args=()
fi
OUT="${OUT:-results/input_graphgps_optimization/${output_stem}${protocol_suffix}}"

# Override for a smoke test or a subset of the frozen manifests, e.g.
# SPLIT_SEEDS="100 101" SINGLE_TARGET=norm_after bash "$0".
read -r -a SPLIT_SEEDS <<< "${SPLIT_SEEDS:-100 101 102 103 104 105 106 107 108 109}"

for seed in "${SPLIT_SEEDS[@]}"; do
 m="$MANIFESTS/$(manifest_name "$seed")"; pre="$OUT/preprocessing/seed$seed"; mkdir -p "$pre" "$OUT/logs"
 [[ -f "$m" ]] || { echo "Missing frozen $SPLIT_PROTOCOL manifest: $m" >&2; exit 2; }
 PYTHONPATH=. "$PYTHON" scripts/diagnostics/build_o13g_structured_features.py --input-csv "$INPUT" --manifest "$m" --raw-output "$OUT/o13g_fifth_structured_features_raw.csv" --lookup-output "$pre/structured.csv" --audit "$pre/feature_audit.json"
 run="$OUT/$run_subdir/${candidate_prefix}_split$seed"; [[ ! -e "$run" ]] || { echo "existing $run" >&2; exit 1; }
 PYTHONPATH=. "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py --config "$CONFIG" --run-dir "$run" --input-csv "$INPUT" --target-set "$TARGET_SET" "${single_target_args[@]}" --split-manifest "$m" --fold "${fold_prefix}$seed" --group B --candidate "${candidate_prefix}_split$seed" --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS --graph-pooling mean --seed 43 --base-lr 0.001 --weight-decay 1e-5 --batch-size 8 --warmup-epochs 10 --early-stop-patience 50 --gt-dropout 0.1 --gt-attn-dropout 0.2 --gps-layers 2 --disable-mordred-features --use-component-aux-features --use-fifth-class-embedding --use-fifth-structured-features --fifth-structured-feature-path "$pre/structured.csv" --execution-max-epochs 300 --include-test 2>&1 | tee "$OUT/logs/seed$seed.log"
done

# The class-restricted protocol has a matching labelled external validation
# table.  Infer only after all requested seeds completed, then score and plot
# the ten-model (or requested-subset) ensemble.  The predictor zeros labels in
# its loader input; labels are read only by the subsequent scoring command.
if [[ -n "$GROUP" ]]; then
    validation_output="$OUT/corresponding_validation_ensemble"
    if [[ -n "$SINGLE_TARGET" ]]; then
        score_key="single_${target_slug}"
    else
        score_key="$TARGET_SET"
    fi
    PYTHONPATH=. "$PYTHON" scripts/diagnostics/predict_o13e_new_validation_ensemble.py \
        --model-root "$OUT" --preprocessing-root "$OUT/preprocessing" \
        --input-csv "$VALIDATION_CSV" --output-root "$validation_output" \
        --target-group "$TARGET_SET" "${single_target_args[@]}" --seeds "${SPLIT_SEEDS[@]}"
    PYTHONPATH=. "$PYTHON" scripts/diagnostics/score_labelled_o12_10seed_ensemble.py \
        --labels-csv "$VALIDATION_CSV" --prediction-dir "$validation_output/$(basename "${VALIDATION_CSV%.csv}")" \
        --output-dir "$validation_output/$(basename "${VALIDATION_CSV%.csv}")/scored_${score_key}" \
        --target-group "$TARGET_SET" "${single_target_args[@]}" \
        --model-label "O13G ${GROUP} ensemble"
    echo "Finished ${GROUP} validation metrics and scatter plots: $validation_output"
fi

#!/usr/bin/env bash
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

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
# The supplied 20260812 CSV is GB18030-encoded.  It is staged as UTF-8 below
# because the runner and PyG loader intentionally read their training CSVs as
# UTF-8 only.
SOURCE_INPUT_CSV="${INPUT_CSV:-datasets_lrx/raw/input/20260812-sum-700.csv}"
MANIFESTS="${MANIFESTS:-results/input_graphgps_optimization/five_split_manifests}"
BASE_CONFIG="${BASE_CONFIG:-results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/source_config.yaml}"
MORDRED="${MORDRED:-results/input_graphgps_optimization/features/mordred11_train_standardized.csv}"
# A new root keeps this fresh-cache rerun separate from all historical trial-I
# checkpoints and their processed PyG data.
RUNS_ROOT="${RUNS_ROOT:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_warmup10}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
SPLIT_SEEDS=(100 101 102 103 104 105 106 107 108 109)
# Train core4 by default.  Use TRAIN_TARGET_GROUPS="core4 norm2" to run both
# target groups in one invocation, or TRAIN_TARGET_GROUPS=norm2 for norm2 only.
read -r -a TRAIN_TARGET_GROUPS <<< "${TRAIN_TARGET_GROUPS:-core4 norm2}"

for path in "$SOURCE_INPUT_CSV" "$BASE_CONFIG" "$MORDRED"; do
    if [[ ! -f "$path" ]]; then
        echo "Required file is missing: $path" >&2
        exit 2
    fi
done
if [[ ! -d "$MANIFESTS" ]]; then
    echo "Fixed split-manifest directory is missing: $MANIFESTS" >&2
    exit 2
fi

mkdir -p "$RUNS_ROOT/logs"

# Always stage the selected input as UTF-8 inside this new run root before any
# cache is built.  This avoids the GB18030 source encoding reaching pandas in
# the loader, and makes the exact consumed CSV an experiment artifact.
STAGED_INPUT_CSV="$RUNS_ROOT/staging/$(basename "${SOURCE_INPUT_CSV%.csv}")_utf8.csv"
mkdir -p "$(dirname "$STAGED_INPUT_CSV")"
"$PYTHON" - "$SOURCE_INPUT_CSV" "$STAGED_INPUT_CSV" <<'PY'
from pathlib import Path
import sys

import pandas as pd

source, destination = map(Path, sys.argv[1:])
for encoding in ("utf-8-sig", "utf-8", "gb18030"):
    try:
        frame = pd.read_csv(source, encoding=encoding, dtype={"ID": str})
        break
    except UnicodeDecodeError:
        continue
else:
    raise UnicodeError(f"Unable to decode training CSV as UTF-8 or GB18030: {source}")

required = {
    "ID", "IL_SMILE", "HL_SMILE", "Chol_SMILE", "PEG_SMILE", "Fifth_SMILE",
    "mol%_IL", "mol%_HL", "mol%_Chol", "mol%_PEG", "mol%_Fifth",
    "EE_before", "EE_after", "Aerosolization_Efficiency", "mRNA_Recovery_Efficiency",
    "Norm_before", "Norm_after",
}
missing = sorted(required.difference(frame.columns))
if missing:
    raise ValueError(f"Training CSV is missing required columns: {missing}")
if len(frame) != 700 or frame["ID"].isna().any() or frame["ID"].duplicated().any():
    raise ValueError("Trial I requires exactly 700 rows with unique, non-null ID values.")
destination.parent.mkdir(parents=True, exist_ok=True)
frame.to_csv(destination, index=False, encoding="utf-8")
print(f"Staged {len(frame)} rows from {source} ({encoding}) to {destination}", flush=True)
PY
INPUT_CSV="$STAGED_INPUT_CSV"

for target_group in "${TRAIN_TARGET_GROUPS[@]}"; do
    if [[ "$target_group" != "core4" && "$target_group" != "norm2" ]]; then
        echo "Unsupported TRAIN_TARGET_GROUPS entry: $target_group; use core4 and/or norm2." >&2
        exit 2
    fi
done

run_group() {
    local target_group="$1"
    local candidate_prefix="$2"
    local split_seed manifest run_dir candidate output_activation
    output_activation="identity"
    if [[ "$target_group" == "core4" ]]; then
        output_activation="sigmoid"
    fi

    for split_seed in "${SPLIT_SEEDS[@]}"; do
        manifest="$MANIFESTS/split_manifest_seed${split_seed}.csv"
        run_dir="$RUNS_ROOT/$target_group/O12_split${split_seed}"
        candidate="${candidate_prefix}${split_seed}"

        if [[ ! -f "$manifest" ]]; then
            echo "Missing fixed split manifest: $manifest" >&2
            exit 1
        fi
        if [[ -e "$run_dir" ]]; then
            if [[ -f "$run_dir/summary.json" && \
                  -f "$run_dir/predictions.csv" && \
                  -f "$run_dir/checkpoints/selected_best.pt" ]]; then
                echo "Skipping completed fresh-cache trial-I ${target_group} split ${split_seed}"
                continue
            fi
            echo "Refusing to resume/reuse incomplete run: $run_dir" >&2
            echo "Choose a new RUNS_ROOT for a fresh processed-cache rebuild." >&2
            exit 1
        fi

        echo "Training fresh-cache trial-I ${target_group} split ${split_seed}"
        # --tqdm-progress is an epoch-level bar for this one seed.  ``tee``
        # keeps it visible in the terminal while retaining a complete log.
        "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
            --config "$BASE_CONFIG" \
            --run-dir "$run_dir" \
            --input-csv "$INPUT_CSV" \
            --target-set "$target_group" \
            --split-manifest "$manifest" \
            --fold "split${split_seed}" --group B --candidate "$candidate" \
            --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
            --output-activation "$output_activation" \
            --seed 43 --base-lr 0.001 --weight-decay 1e-5 --batch-size 8 \
            --warmup-epochs 10 --early-stop-patience "$EARLY_STOP_PATIENCE" \
            --gt-dropout 0.1 --gt-attn-dropout 0.2 --gps-layers 2 \
            --use-mordred-features --mordred-feature-dim 11 \
            --mordred-feature-path "$MORDRED" --use-component-aux-features \
            --execution-max-epochs "$MAX_EPOCHS" --include-test --tqdm-progress \
            2>&1 | tee "$RUNS_ROOT/logs/O12_${target_group}_split${split_seed}.log"
    done
}

# Keep training sequential: this avoids competing for the same local GPU and
# exactly preserves the deterministic per-run setup of experiment I.
for target_group in "${TRAIN_TARGET_GROUPS[@]}"; do
    case "$target_group" in
        core4) run_group core4 O12S ;;
        norm2) run_group norm2 O12N ;;
    esac
done

# Explicit command boundary before the post-training evaluation section.  This
# prevents a malformed continuation in an edited copy of the training command
# from being carried into the following shell code.
:

# Freeze the finished selected-best checkpoints and evaluate each once on its
# own fixed validation and test membership.  The output includes per-target
# MAE/R2/Pearson/Spearman tables and 10-seed means/variances.  A target group
# is evaluated only when every requested split has a finished selected-best
# checkpoint; this lets a completed core4 run be evaluated while norm2 is
# still absent or intentionally deferred.
group_is_complete() {
    local target_group="$1"
    local split_seed run_dir
    for split_seed in "${SPLIT_SEEDS[@]}"; do
        run_dir="$RUNS_ROOT/$target_group/O12_split${split_seed}"
        if [[ ! -f "$run_dir/summary.json" || \
              ! -f "$run_dir/predictions.csv" || \
              ! -f "$run_dir/checkpoints/selected_best.pt" ]]; then
            return 1
        fi
    done
}

completed_groups=()
for target_group in core4 norm2; do
    if group_is_complete "$target_group"; then
        completed_groups+=("$target_group")
    else
        echo "Skipping ${target_group} evaluation: one or more 100-109 checkpoints are incomplete."
    fi
done
if (( ${#completed_groups[@]} )); then
    "$PYTHON" scripts/diagnostics/evaluate_o12_10seed_corresponding_splits.py \
        --model-root "$RUNS_ROOT" \
        --manifest-root "$MANIFESTS" \
        --output-dir "$RUNS_ROOT/corresponding_split_single_inference" \
        --target-groups "${completed_groups[@]}"
else
    echo "No complete target group is available for frozen validation/test evaluation."
fi

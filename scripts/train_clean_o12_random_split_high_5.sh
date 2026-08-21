#!/usr/bin/env bash
set -euo pipefail

# Clean O12 single-task experiment:
#   OneHotEmbedGPS + raw component identities/ratios + Fifth molecular graph
#   NO Mordred / Morgan-RDKit auxiliary / mechanistic / semantic /
#      structured / Fifth identity/class engineered feature branches.
#
# Latest sigmoid target weighting defaults:
#   threshold = 1.0
#   low       = 0.1
#   high      = 5.0
#   tau       = 0.15
#
# Override examples:
#   TARGETS="Norm_before" bash scripts/train_O12_clean_sigmoid_high_5.sh
#   NORM_WEIGHT_HIGH=10 bash scripts/train_O12_clean_sigmoid_high_5.sh

PY="${PY:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"

BASE="${BASE:-results/input_graphgps_optimization/experiments/O12_onehot_graphgps_only.yaml}"

INPUT="${INPUT:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline/staging/20260812-sum-700_utf8.csv}"

MANIFESTS="${MANIFESTS:-results/input_graphgps_optimization/five_split_manifests}"

OUT="${OUT:-results/input_graphgps_optimization/o12_clean_sigmoid_high_5}"

TARGETS="${TARGETS:-Norm_before Norm_after}"
read -r -a TARGET_LIST <<< "$TARGETS"

SPLIT_SEEDS="${SPLIT_SEEDS:-100 101 102 103 104 105 106 107 108 109}"
read -r -a SEED_LIST <<< "$SPLIT_SEEDS"

NORM_THRESHOLD="${NORM_THRESHOLD:-1.0}"
NORM_WEIGHT_LOW="${NORM_WEIGHT_LOW:-0.1}"
NORM_WEIGHT_HIGH="${NORM_WEIGHT_HIGH:-5.0}"
NORM_WEIGHT_TAU="${NORM_WEIGHT_TAU:-2.0}"

RUNNER="scripts/diagnostics/run_fusion_head_experiment.py"

[[ -x "$PY" ]] || {
    echo "Python executable not found or not executable: $PY" >&2
    exit 2
}
[[ -f "$RUNNER" ]] || {
    echo "Runner missing: $RUNNER" >&2
    exit 2
}
[[ -f "$BASE" ]] || {
    echo "Clean YAML missing: $BASE" >&2
    exit 2
}
[[ -f "$INPUT" ]] || {
    echo "UTF-8 training CSV missing: $INPUT" >&2
    exit 2
}

# Ensure the local runner actually contains the sigmoid-weighting patch.
"$PY" "$RUNNER" --help 2>&1 | grep -q -- "--enable-norm-sigmoid-weighting" || {
    echo "The local runner does not support --enable-norm-sigmoid-weighting." >&2
    exit 2
}
"$PY" "$RUNNER" --help 2>&1 | grep -q -- "--norm-weight-tau" || {
    echo "The local runner does not support --norm-weight-tau." >&2
    exit 2
}

# Preflight:
# 1) INPUT must be UTF-8-readable.
# 2) YAML must have every engineered feature branch disabled.
# 3) All requested manifests must contain exactly the same IDs as INPUT.
"$PY" - "$BASE" "$INPUT" "$MANIFESTS" "${SEED_LIST[@]}" <<'PY'
import sys
from pathlib import Path

import pandas as pd
import yaml

base = Path(sys.argv[1])
input_csv = Path(sys.argv[2])
manifest_root = Path(sys.argv[3])
seeds = [int(x) for x in sys.argv[4:]]

cfg = yaml.safe_load(base.read_text(encoding="utf-8"))

disabled = {
    "use_component_aux_features": False,
    "use_mordred_features": False,
    "use_fifth_mechanistic_descriptors": False,
    "use_fifth_semantic_features": False,
    "use_fifth_structured_features": False,
    "use_fifth_identity_embedding": False,
    "use_fifth_class_embedding": False,
    "use_fifth_ratio_modulation": False,
}
for key, expected in disabled.items():
    actual = bool(cfg.get(key, False))
    if actual != expected:
        raise SystemExit(f"Clean YAML check failed: {key}={actual}, expected {expected}")

model = cfg.get("model", {})
if model.get("type") != "OneHotEmbedGPS":
    raise SystemExit(
        f"Clean YAML check failed: model.type={model.get('type')!r}, "
        "expected 'OneHotEmbedGPS'"
    )
if bool(model.get("ratio_polynomial_features", False)):
    raise SystemExit("Clean YAML check failed: ratio_polynomial_features is enabled.")
if bool(model.get("fifth_only_fusion", False)):
    raise SystemExit("Clean YAML check failed: fifth_only_fusion is enabled.")

# Explicit UTF-8 read. This is intended to fail here rather than after run dirs
# have already been created.
data = pd.read_csv(input_csv, dtype={"ID": str}, encoding="utf-8")
if "ID" not in data.columns:
    raise SystemExit("Training CSV has no ID column.")
if data["ID"].duplicated().any():
    raise SystemExit("Training CSV contains duplicate IDs.")

data_ids = set(data["ID"].astype(str))

for seed in seeds:
    manifest = manifest_root / f"split_manifest_seed{seed}.csv"
    if not manifest.is_file():
        raise SystemExit(f"Missing split manifest: {manifest}")
    m = pd.read_csv(manifest, dtype={"sample_id": str}, encoding="utf-8")
    required = {"sample_id", "split"}
    missing = required.difference(m.columns)
    if missing:
        raise SystemExit(f"{manifest} misses columns: {sorted(missing)}")
    if m["sample_id"].duplicated().any():
        raise SystemExit(f"{manifest} contains duplicate sample_id values.")
    manifest_ids = set(m["sample_id"].astype(str))
    if data_ids != manifest_ids:
        only_data = sorted(data_ids - manifest_ids)[:5]
        only_manifest = sorted(manifest_ids - data_ids)[:5]
        raise SystemExit(
            f"ID mismatch for seed {seed}: "
            f"only_data={only_data}, only_manifest={only_manifest}"
        )

print(
    f"Preflight PASS: rows={len(data)}, "
    f"seeds={seeds[0]}..{seeds[-1]}, "
    "descriptor branches disabled."
)
PY

mkdir -p "$OUT/logs"

for TARGET in "${TARGET_LIST[@]}"; do
    case "$TARGET" in
        Norm_before)
            target_slug="norm_before"
            ;;
        Norm_after)
            target_slug="norm_after"
            ;;
        *)
            echo "Unsupported clean sigmoid target: $TARGET" >&2
            exit 2
            ;;
    esac

    for SPLIT_SEED in "${SEED_LIST[@]}"; do
        manifest="$MANIFESTS/split_manifest_seed${SPLIT_SEED}.csv"
        run_dir="$OUT/$target_slug/O12Clean_${TARGET}_split${SPLIT_SEED}"
        log="$OUT/logs/O12Clean_${TARGET}_split${SPLIT_SEED}.log"
        candidate="O12Clean_${TARGET}_W${NORM_WEIGHT_HIGH}_S${SPLIT_SEED}"

        # Completed run: do not overwrite it.
        if [[ -f "$run_dir/summary.json" && -f "$run_dir/predictions.csv" ]]; then
            echo "Skipping completed run: $run_dir"
            continue
        fi

        lifecycle_args=()
        if [[ -f "$run_dir/resume_state.pt" ]]; then
            lifecycle_args+=(--resume)
            echo "Resuming: $TARGET seed=$SPLIT_SEED"
        elif [[ -d "$run_dir" ]]; then
            # Handles directories left by pre-epoch failures such as a bad
            # vocabulary path/encoding without requiring manual rm -rf.
            lifecycle_args+=(--restart-incomplete)
            echo "Restarting incomplete run: $TARGET seed=$SPLIT_SEED"
        else
            echo "Starting: $TARGET seed=$SPLIT_SEED"
        fi

        "$PY" "$RUNNER" \
            --config "$BASE" \
            --input-csv "$INPUT" \
            --component-vocab-source "$INPUT" \
            --run-dir "$run_dir" \
            --target-set norm2 \
            --single-target "$TARGET" \
            --split-manifest "$manifest" \
            --fold "split${SPLIT_SEED}" \
            --group B \
            --candidate "$candidate" \
            --fusion-type concat_mlp \
            --head-type baseline \
            --model-type OneHotEmbedGPS \
            --seed 43 \
            --base-lr 0.001 \
            --weight-decay 1e-5 \
            --gt-dropout 0.1 \
            --gt-attn-dropout 0.2 \
            --disable-mordred-features \
            --execution-max-epochs 300 \
            --include-test \
            --enable-norm-sigmoid-weighting \
            --norm-threshold "$NORM_THRESHOLD" \
            --norm-weight-low "$NORM_WEIGHT_LOW" \
            --norm-weight-high "$NORM_WEIGHT_HIGH" \
            --norm-weight-tau "$NORM_WEIGHT_TAU" \
            "${lifecycle_args[@]}" \
            2>&1 | tee "$log"
    done
done

echo
echo "All requested O12-clean runs finished."
echo "Output root: $OUT"
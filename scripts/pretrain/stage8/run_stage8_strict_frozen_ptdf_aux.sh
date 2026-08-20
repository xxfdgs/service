#!/usr/bin/env bash
# Stage 8 strict No-Mordred screening: frozen PT-DF structural prior.
#
# Only P3 is trained here.  It is compared against the independently trained
# strict No-Mordred P0/P1/P2 controls from Stage 6.  The architecture is the
# existing Stage8A topology: a random/trainable task Comp5 encoder plus a
# distinct Stage-4 PT-DF Comp5 encoder frozen in eval mode.
#
# Run:
#   bash scripts/pretrain/stage8/run_stage8_strict_frozen_ptdf_aux.sh
#
# Preflight only:
#   PREFLIGHT_ONLY=1 bash scripts/pretrain/stage8/run_stage8_strict_frozen_ptdf_aux.sh
#
# Construction/transfer/optimizer audit only:
#   SMOKE_AUDIT_ONLY=1 bash scripts/pretrain/stage8/run_stage8_strict_frozen_ptdf_aux.sh

set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
RUNNER="${RUNNER:-scripts/diagnostics/run_fusion_head_experiment.py}"

O12_BASELINE_ROOT="${O12_BASELINE_ROOT:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
INPUT_CSV="${INPUT_CSV:-$O12_BASELINE_ROOT/staging/20260812-sum-700_utf8.csv}"
BASE_CONFIG="${BASE_CONFIG:-$O12_BASELINE_ROOT/core4/O12_split100/source_config.yaml}"
OOD_MANIFESTS="${OOD_MANIFESTS:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests}"

STAGE4_ROOT="${STAGE4_ROOT:-results/fifth_pretraining/stage4_graphgps_pretraining}"
PT_DF="${PT_DF:-$STAGE4_ROOT/PT_DF/checkpoints/best_comp5_encoder_state_dict.pt}"

# This root is deliberately distinct from both Stage8A legacy-Mordred output
# and the strict No-Mordred P0/P1/P2 control root.
TRANSFER_ROOT="${TRANSFER_ROOT:-results/fifth_pretraining/stage8_strict_no_mordred_fifth_ood_membership_fixed}"
LABEL="${LABEL:-P3_PT_DF_FrozenAux_NoMordred}"
CONTROLS_ROOT="${CONTROLS_ROOT:-results/fifth_pretraining/stage6_strict_no_mordred_fifth_ood_baseline}"
P0_LABEL="${P0_LABEL:-P0_random_strict_no_mordred}"

BASE_LR="${BASE_LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-50}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
TRAIN_RNG_SEED="${TRAIN_RNG_SEED:-43}"
SPLIT_SEEDS="${SPLIT_SEEDS:-100 101 102}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
SMOKE_AUDIT_ONLY="${SMOKE_AUDIT_ONLY:-0}"
RUN_SMOKE_AUDIT="${RUN_SMOKE_AUDIT:-1}"

read -r -a SPLITS <<< "$SPLIT_SEEDS"

[[ -x "$PYTHON" ]] || {
    echo "Configured Python is not executable: $PYTHON" >&2
    exit 2
}

for path in "$RUNNER" "$INPUT_CSV" "$BASE_CONFIG" "$PT_DF"; do
    [[ -f "$path" ]] || {
        echo "Missing locked Stage-8 strict input: $path" >&2
        exit 2
    }
done
for split_seed in "${SPLITS[@]}"; do
    manifest="$OOD_MANIFESTS/fifth_identity_manifest_seed${split_seed}.csv"
    [[ -f "$manifest" ]] || {
        echo "Missing frozen Fifth-OOD manifest: $manifest" >&2
        exit 2
    }
    p0_run_dir="$CONTROLS_ROOT/$P0_LABEL/split${split_seed}"
    [[ -f "$p0_run_dir/predictions.csv" ]] || {
        echo "Missing strict P0 membership reference: $p0_run_dir/predictions.csv" >&2
        exit 2
    }
done

"$PYTHON" - <<'PY'
from pathlib import Path

runner = Path("scripts/diagnostics/run_fusion_head_experiment.py")
text = runner.read_text(encoding="utf-8")
model = Path("graphgps/network/onehot_embed_gps.py").read_text(encoding="utf-8")
required_runner = [
    "--disable-mordred-features",
    "--frozen-comp5-aux-checkpoint",
    "--architecture-audit-only",
    "--require-membership-count",
    "--membership-reference-run-dir",
    "--require-fresh-cache",
    "frozen_comp5_aux_initialization.json",
    "Frozen auxiliary resume topology mismatch",
    "Optimizer parameter set does not exactly equal",
]
missing = [marker for marker in required_runner if marker not in text]
if missing:
    raise SystemExit("Stage-8 strict runner prerequisites missing: " + ", ".join(missing))
for marker in (
    "frozen_comp5_aux_encoder",
    "def train(self, mode=True):",
    "frozen_input = data5.clone()",
    "fusion_parts.insert(1, emb5_frozen)",
):
    if marker not in model:
        raise SystemExit(f"Stage-8 model prerequisite missing: {marker}")
print("[Stage8 strict] source preflight: PASS")
PY

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "[Stage8 strict] PREFLIGHT_ONLY=1; no cache/model/training was started."
    exit 0
fi

run_common() {
    local split_seed="$1"
    local run_dir="$2"
    shift 2
    local manifest="$OOD_MANIFESTS/fifth_identity_manifest_seed${split_seed}.csv"

    "$PYTHON" -u "$RUNNER" \
        --config "$BASE_CONFIG" \
        --run-dir "$run_dir" \
        --input-csv "$INPUT_CSV" \
        --target-set norm2 \
        --single-target Norm_before \
        --split-manifest "$manifest" \
        --fold "fifth_identity_ood_split${split_seed}" \
        --group B \
        --candidate "Stage8FrozenPTDFAux_StrictNoMordred" \
        --fusion-type concat_mlp \
        --head-type baseline \
        --model-type OneHotEmbedGPS \
        --graph-pooling mean \
        --use-fifth-class-embedding \
        --output-activation identity \
        --seed "$TRAIN_RNG_SEED" \
        --base-lr "$BASE_LR" \
        --weight-decay "$WEIGHT_DECAY" \
        --batch-size "$BATCH_SIZE" \
        --warmup-epochs "$WARMUP_EPOCHS" \
        --early-stop-patience "$EARLY_STOP_PATIENCE" \
        --gt-dropout 0.1 \
        --gt-attn-dropout 0.2 \
        --gps-layers 2 \
        --disable-mordred-features \
        --require-membership-count 700 \
        --require-fresh-cache \
        --membership-reference-run-dir "$CONTROLS_ROOT/$P0_LABEL/split${split_seed}" \
        --use-component-aux-features \
        --execution-max-epochs "$MAX_EPOCHS" \
        --include-test \
        --comp5-pretrain-label "${LABEL}_task_random" \
        --frozen-comp5-aux-checkpoint "$PT_DF" \
        --frozen-comp5-aux-label "Stage4_PT_DF_frozen_structural_prior" \
        "$@"
}

run_smoke_audit() {
    local audit_root
    audit_root="$(mktemp -d /tmp/stage8-strict-no-mordred-audit.XXXXXX)"
    trap 'rm -rf -- "$audit_root"' RETURN

    echo "[Stage8 strict] running split100 construction/transfer/optimizer audit..."
    run_common 100 "$audit_root/split100" --architecture-audit-only
    "$PYTHON" - "$audit_root/split100" "$PT_DF" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_dir, checkpoint = map(Path, sys.argv[1:])
audit = json.loads((run_dir / "architecture_audit.json").read_text(encoding="utf-8"))
settings = json.loads((run_dir / "run_settings.json").read_text(encoding="utf-8"))
effective = (run_dir / "effective_config.yaml").read_text(encoding="utf-8")
frozen = audit["frozen_comp5_aux_initialization"]
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

assert audit["status"] == "PASS"
assert settings["use_mordred_features"] is False
assert settings["mordred_feature_path"] == ""
assert settings["mordred_feature_dim"] == 0
assert settings["strict_no_mordred"] is True
assert settings["frozen_comp5_aux_checkpoint_sha256"] == digest
assert settings["frozen_comp5_aux_initialization"]["checkpoint_sha256"] == digest
assert frozen["checkpoint_sha256"] == digest
assert frozen["strict_transfer_report"]["strict"] is True
assert frozen["trainable_parameter_count"] == 0
assert frozen["task_comp5_trainable_parameter_count"] > 0
assert frozen["optimizer_includes_frozen_parameters"] is False
assert frozen["optimizer_exact_trainable_partition"] is True
assert frozen["frozen_training_after_model_train"] is False
assert frozen["topology"]["task_and_frozen_encoder_distinct"] is True
assert "frozen_comp5_aux_parameter_count: 89008" in effective
assert "frozen_comp5_aux_trainable_parameter_count: 0" in effective
assert "frozen_comp5_aux_optimizer_includes_parameters: false" in effective
print("[Stage8 strict] split100 smoke audit: PASS")
PY
}

if [[ "$RUN_SMOKE_AUDIT" == "1" || "$SMOKE_AUDIT_ONLY" == "1" ]]; then
    run_smoke_audit
fi
if [[ "$SMOKE_AUDIT_ONLY" == "1" ]]; then
    echo "[Stage8 strict] SMOKE_AUDIT_ONLY=1; formal training was not started."
    exit 0
fi

run_is_complete_and_strict() {
    local run_dir="$1"
    [[ -f "$run_dir/summary.json" ]] || return 1
    [[ -f "$run_dir/predictions.csv" ]] || return 1
    [[ -f "$run_dir/checkpoints/selected_best.pt" ]] || return 1
    [[ -f "$run_dir/comp5_initialization.json" ]] || return 1
    [[ -f "$run_dir/frozen_comp5_aux_initialization.json" ]] || return 1
    [[ -f "$run_dir/membership_audit.json" ]] || return 1
    [[ -f "$run_dir/membership_audit.csv" ]] || return 1
    [[ -f "$run_dir/run_settings.json" ]] || return 1
    [[ -f "$run_dir/effective_config.yaml" ]] || return 1

    "$PYTHON" - "$run_dir" "$LABEL" "$PT_DF" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_dir, label, checkpoint = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
settings = json.loads((run_dir / "run_settings.json").read_text(encoding="utf-8"))
task = json.loads((run_dir / "comp5_initialization.json").read_text(encoding="utf-8"))
frozen = json.loads((run_dir / "frozen_comp5_aux_initialization.json").read_text(encoding="utf-8"))
membership = json.loads((run_dir / "membership_audit.json").read_text(encoding="utf-8"))
checkpoint_payload = __import__("torch").load(
    run_dir / "checkpoints" / "selected_best.pt", map_location="cpu", weights_only=False
)
expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

assert settings["use_mordred_features"] is False
assert settings["mordred_feature_path"] == ""
assert settings["mordred_feature_dim"] == 0
assert settings["strict_no_mordred"] is True
assert settings["frozen_comp5_aux_enable"] is True
assert settings["frozen_comp5_aux_checkpoint_sha256"] == expected_sha
assert settings["frozen_comp5_aux_initialization"]["checkpoint_sha256"] == expected_sha
assert task["label"] == f"{label}_task_random" and task["mode"] == "random"
assert frozen["enabled"] is True and frozen["checkpoint_sha256"] == expected_sha
assert frozen["strict_transfer_report"]["strict"] is True
assert frozen["trainable_parameter_count"] == 0
assert frozen["task_comp5_trainable_parameter_count"] > 0
assert frozen["optimizer_includes_frozen_parameters"] is False
assert frozen["optimizer_exact_trainable_partition"] is True
assert frozen["frozen_training_after_model_train"] is False
assert frozen["topology"]["task_and_frozen_encoder_distinct"] is True
meta = checkpoint_payload["frozen_comp5_aux_initialization"]
assert meta["checkpoint_sha256"] == expected_sha
assert checkpoint_payload["use_mordred_features"] is False
assert membership["status"] == "PASS"
assert membership["requirements"]["required_union_count"] == 700
assert membership["requirements"]["reference_partitions_pass"] is True
assert membership["prediction_export"]["status"] == "PASS"
PY
}

for split_seed in "${SPLITS[@]}"; do
    run_dir="$TRANSFER_ROOT/$LABEL/split${split_seed}"
    log_dir="$TRANSFER_ROOT/$LABEL/logs"
    log="$log_dir/split${split_seed}.log"
    mkdir -p "$log_dir"

    if [[ -e "$run_dir" ]]; then
        if run_is_complete_and_strict "$run_dir"; then
            echo "[skip] completed strict Stage8 P3: split${split_seed}"
            continue
        fi
        echo "Refusing existing non-complete or non-strict Stage8 directory:" >&2
        echo "  $run_dir" >&2
        echo "It will not be reused or overwritten." >&2
        exit 1
    fi

    echo
    echo "================================================================================"
    echo "Stage8 Strict No-Mordred | $LABEL | Fifth-OOD split $split_seed"
    echo "task branch=random/trainable | prior=PT-DF/frozen | LR=$BASE_LR | RNG=$TRAIN_RNG_SEED"
    echo "Mordred disabled; component aux, Fifth_class, ratio, fusion/head retained"
    echo "================================================================================"
    run_common "$split_seed" "$run_dir" 2>&1 | tee "$log"
done

echo
echo "Stage8 strict No-Mordred screening complete."
echo "Label : $LABEL"
echo "Splits: ${SPLITS[*]}"
echo "Root  : $TRANSFER_ROOT/$LABEL"

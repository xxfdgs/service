#!/usr/bin/env bash
# Strict No-Mordred Fifth-identity OOD baseline.
#
# This is an intentionally separate reproduction of the Stage-5/6 Norm_before
# full-fine-tuning protocol.  It retains the O13-D/Stage-5 model topology,
# component auxiliary features, Fifth_class embedding, ratio/fusion path, and
# validation-selected checkpoint logic, but explicitly disables Mordred in the
# runner.  No Mordred lookup is accepted or read by this script.
#
# Models (all use one ordinary LR for every trainable parameter):
#   P0_random_strict_no_mordred : random Comp5GraphEncoder
#   P1_PT_D_strict_no_mordred   : strict Stage-4 PT-D transfer, full FT
#   P2_PT_DF_strict_no_mordred  : strict Stage-4 PT-DF transfer, full FT
#
# Run:
#   bash scripts/pretrain/stage6/run_strict_no_mordred_fifth_ood_baseline.sh
#
# Safe preflight without training:
#   PREFLIGHT_ONLY=1 bash scripts/pretrain/stage6/run_strict_no_mordred_fifth_ood_baseline.sh

set -euo pipefail

cd "$(dirname "$0")/../../.."

# Keep this overridable for another host, while avoiding a silent fallback to
# a system interpreter that lacks the project PyG/RDKit environment.
PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
RUNNER="${RUNNER:-scripts/diagnostics/run_fusion_head_experiment.py}"

O12_BASELINE_ROOT="${O12_BASELINE_ROOT:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
INPUT_CSV="${INPUT_CSV:-$O12_BASELINE_ROOT/staging/20260812-sum-700_utf8.csv}"
BASE_CONFIG="${BASE_CONFIG:-$O12_BASELINE_ROOT/core4/O12_split100/source_config.yaml}"
OOD_MANIFESTS="${OOD_MANIFESTS:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests}"

STAGE4_ROOT="${STAGE4_ROOT:-results/fifth_pretraining/stage4_graphgps_pretraining}"
PT_D="${PT_D:-$STAGE4_ROOT/PT_D/checkpoints/best_comp5_encoder_state_dict.pt}"
PT_DF="${PT_DF:-$STAGE4_ROOT/PT_DF/checkpoints/best_comp5_encoder_state_dict.pt}"

# Do not point this at stage5_downstream_transfer: keeping a distinct root and
# labels makes accidental mixing with the legacy global-Mordred runs impossible.
TRANSFER_ROOT="${TRANSFER_ROOT:-results/fifth_pretraining/stage6_strict_no_mordred_fifth_ood_baseline}"

BASE_LR="${BASE_LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-50}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
TRAIN_RNG_SEED="${TRAIN_RNG_SEED:-43}"
SPLIT_SEEDS="${SPLIT_SEEDS:-100 101 102 103 104 105 106 107 108 109}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

P0_LABEL="P0_random_strict_no_mordred"
P1_LABEL="P1_PT_D_strict_no_mordred"
P2_LABEL="P2_PT_DF_strict_no_mordred"

read -r -a SPLITS <<< "$SPLIT_SEEDS"

[[ -x "$PYTHON" ]] || {
    echo "Configured Python is not executable: $PYTHON" >&2
    exit 2
}

for path in \
    "$RUNNER" \
    "$INPUT_CSV" \
    "$BASE_CONFIG" \
    "$PT_D" \
    "$PT_DF"; do
    [[ -f "$path" ]] || {
        echo "Missing locked strict No-Mordred input: $path" >&2
        exit 2
    }
done

for split_seed in "${SPLITS[@]}"; do
    manifest="$OOD_MANIFESTS/fifth_identity_manifest_seed${split_seed}.csv"
    [[ -f "$manifest" ]] || {
        echo "Missing frozen Fifth-OOD manifest: $manifest" >&2
        exit 2
    }
done

"$PYTHON" - <<'PY'
from pathlib import Path

runner = Path("scripts/diagnostics/run_fusion_head_experiment.py")
text = runner.read_text(encoding="utf-8")
required = [
    "--disable-mordred-features",
    "--comp5-pretrained-checkpoint",
    "--comp5-pretrain-label",
    "load_stage4_comp5_encoder",
    "comp5_initialization.json",
]
missing = [marker for marker in required if marker not in text]
if missing:
    raise SystemExit(
        "Strict No-Mordred runner prerequisites are missing: "
        + ", ".join(missing)
    )
print("[Strict No-Mordred] runner preflight: PASS")
PY

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "[Strict No-Mordred] PREFLIGHT_ONLY=1; training was not started."
    exit 0
fi

run_is_complete_and_strict() {
    local run_dir="$1"
    local expected_label="$2"
    local expected_mode="$3"

    [[ -f "$run_dir/summary.json" ]] || return 1
    [[ -f "$run_dir/predictions.csv" ]] || return 1
    [[ -f "$run_dir/checkpoints/selected_best.pt" ]] || return 1
    [[ -f "$run_dir/comp5_initialization.json" ]] || return 1
    [[ -f "$run_dir/run_settings.json" ]] || return 1
    [[ -f "$run_dir/effective_config.yaml" ]] || return 1

    "$PYTHON" - "$run_dir" "$expected_label" "$expected_mode" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected_label, expected_mode = sys.argv[2:]
settings = json.loads((run_dir / "run_settings.json").read_text(encoding="utf-8"))
init = json.loads((run_dir / "comp5_initialization.json").read_text(encoding="utf-8"))

if settings.get("use_mordred_features") is not False:
    raise SystemExit("run_settings.json does not prove Mordred was disabled")
if settings.get("mordred_feature_path") not in ("", None):
    raise SystemExit("run_settings.json unexpectedly records a Mordred lookup")
if int(settings.get("mordred_feature_dim", -1)) != 0:
    raise SystemExit("run_settings.json unexpectedly records nonzero Mordred dimension")
if init.get("label") != expected_label:
    raise SystemExit("comp5 initialization label does not match this strict baseline")
if init.get("mode") != expected_mode:
    raise SystemExit("comp5 initialization mode does not match this strict baseline")
if expected_mode != "random":
    report = init.get("strict_transfer_report") or {}
    if report.get("strict") is not True or not init.get("checkpoint_sha256"):
        raise SystemExit("pretrained strict-transfer provenance is incomplete")

effective = (run_dir / "effective_config.yaml").read_text(encoding="utf-8")
if "use_mordred_features: false" not in effective:
    raise SystemExit("effective_config.yaml does not record use_mordred_features: false")
PY
}

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
        --candidate "Stage5NormBeforeFullFT_StrictNoMordred" \
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
        --use-component-aux-features \
        --execution-max-epochs "$MAX_EPOCHS" \
        --include-test \
        "$@"
}

run_model_split() {
    local label="$1"
    local split_seed="$2"
    local expected_mode="$3"
    shift 3

    local run_dir="$TRANSFER_ROOT/$label/split${split_seed}"
    local log_dir="$TRANSFER_ROOT/$label/logs"
    local log="$log_dir/split${split_seed}.log"
    mkdir -p "$log_dir"

    if [[ -e "$run_dir" ]]; then
        if run_is_complete_and_strict "$run_dir" "$label" "$expected_mode"; then
            echo "[skip] completed strict No-Mordred run: $label split${split_seed}"
            return
        fi
        echo "Refusing existing non-complete or non-strict run directory:" >&2
        echo "  $run_dir" >&2
        echo "It will not be reused or overwritten. Choose a fresh root or delete it explicitly." >&2
        exit 1
    fi

    echo
    echo "================================================================================"
    echo "Strict No-Mordred baseline | $label | Fifth-OOD split $split_seed"
    echo "training RNG=$TRAIN_RNG_SEED | ordinary full fine-tuning LR=$BASE_LR"
    echo "Mordred: disabled (--disable-mordred-features); component aux retained"
    echo "================================================================================"

    run_common "$split_seed" "$run_dir" "$@" 2>&1 | tee "$log"
}

for split_seed in "${SPLITS[@]}"; do
    run_model_split \
        "$P0_LABEL" "$split_seed" "random" \
        --comp5-pretrain-label "$P0_LABEL"
    run_model_split \
        "$P1_LABEL" "$split_seed" "stage4_pretrained_full_finetune" \
        --comp5-pretrained-checkpoint "$PT_D" \
        --comp5-pretrain-label "$P1_LABEL"
    run_model_split \
        "$P2_LABEL" "$split_seed" "stage4_pretrained_full_finetune" \
        --comp5-pretrained-checkpoint "$PT_DF" \
        --comp5-pretrain-label "$P2_LABEL"
done

echo
echo "================================================================================"
echo "Strict No-Mordred Fifth-identity OOD baseline complete."
echo "Models : $P0_LABEL | $P1_LABEL | $P2_LABEL"
echo "Splits : ${SPLITS[*]}"
echo "Root   : $TRANSFER_ROOT"
echo "================================================================================"

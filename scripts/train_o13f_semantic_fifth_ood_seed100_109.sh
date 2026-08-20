#!/usr/bin/env bash
# O13-F: O13-C mean pooling + Fifth chemistry-semantic branch, Fifth-OOD only.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
O12_ROOT="${O12_ROOT:-results/input_graphgps_optimization/o12_input_700_multitasks_lr0001_sigmoid_core4_ratiofix_20260812_freshcache_baseline}"
INPUT_CSV="${INPUT_CSV:-$O12_ROOT/staging/20260812-sum-700_utf8.csv}"
BASE_CONFIG="${BASE_CONFIG:-$O12_ROOT/core4/O12_split100/source_config.yaml}"
MANIFEST_ROOT="${MANIFEST_ROOT:-results/input_graphgps_optimization/o12_fifth_identity_ood_seed100_109/fifth_identity_manifests}"
RAW_MORDRED="${RAW_MORDRED:-results/deduplicated_rebaseline/artifacts/mordred_11_lookup.csv}"
ROOT_OUT="${ROOT_OUT:-results/input_graphgps_optimization/o13f_semantic_features}"
RAW_SEMANTIC="$ROOT_OUT/o13f_fifth_semantic_features_raw.csv"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
SPLIT_SEEDS=(100 101 102 103 104 105 106 107 108 109)

for file in "$INPUT_CSV" "$BASE_CONFIG" "$RAW_MORDRED" "$MANIFEST_ROOT/protocol.json"; do
  [[ -f "$file" ]] || { echo "Missing locked input: $file" >&2; exit 2; }
done
PYTHONPATH=. "$PYTHON" scripts/diagnostics/build_o13f_raw_fifth_semantic_lookup.py \
  --input-csv "$INPUT_CSV" --output "$RAW_SEMANTIC" --audit "$ROOT_OUT/o13f_semantic_feature_audit.json"

for seed in "${SPLIT_SEEDS[@]}"; do
  manifest="$MANIFEST_ROOT/fifth_identity_manifest_seed${seed}.csv"
  pre="$ROOT_OUT/preprocessing/seed${seed}"
  mordred="$pre/mordred11_all_components_train_only.csv"
  semantic="$pre/fifth_semantic_train_only.csv"
  mkdir -p "$pre" "$ROOT_OUT/logs"
  PYTHONPATH=. "$PYTHON" scripts/diagnostics/build_o13_train_only_mordred11_lookup.py \
    --input-csv "$INPUT_CSV" --manifest "$manifest" --raw-lookup "$RAW_MORDRED" --output "$mordred"
  PYTHONPATH=. "$PYTHON" scripts/diagnostics/build_o13f_train_only_semantic_lookup.py \
    --input-csv "$INPUT_CSV" --manifest "$manifest" --raw-lookup "$RAW_SEMANTIC" --output "$semantic"
  dim=$(PYTHONPATH=. "$PYTHON" -c "import json; print(json.load(open('${semantic%.csv}.json'))['feature_dim'])")
  for group in core4 norm2; do
    activation=identity; [[ "$group" == core4 ]] && activation=sigmoid
    run="$ROOT_OUT/o13f_strict_train_only_scaling/$group/O12_split${seed}"
    [[ ! -e "$run" ]] || { echo "Refusing existing run directory: $run" >&2; exit 1; }
    PYTHONPATH=. "$PYTHON" scripts/diagnostics/run_fusion_head_experiment.py \
      --config "$BASE_CONFIG" --run-dir "$run" --input-csv "$INPUT_CSV" --target-set "$group" \
      --split-manifest "$manifest" --fold "fifth_identity_ood_split${seed}" --group B \
      --candidate "O13F_${group}_seed${seed}" --fusion-type concat_mlp --head-type baseline \
      --model-type OneHotEmbedGPS --graph-pooling mean --output-activation "$activation" --seed 43 \
      --base-lr 0.001 --weight-decay 1e-5 --batch-size 8 --warmup-epochs 50 \
      --early-stop-patience "$EARLY_STOP_PATIENCE" --gt-dropout 0.1 --gt-attn-dropout 0.2 \
      --gps-layers 2 --use-mordred-features --mordred-feature-dim 11 --mordred-feature-path "$mordred" \
      --use-component-aux-features --use-fifth-semantic-features \
      --fifth-semantic-feature-path "$semantic" --fifth-semantic-feature-dim "$dim" \
      --execution-max-epochs "$MAX_EPOCHS" --include-test 2>&1 | tee "$ROOT_OUT/logs/seed${seed}_${group}.log"
  done
done

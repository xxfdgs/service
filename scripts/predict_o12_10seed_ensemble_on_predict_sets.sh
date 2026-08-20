#!/usr/bin/env bash
# Predict the three formulation tables with both ten-checkpoint O12 groups.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-results/input_graphgps_optimization/O12-10-seeds-prediction-models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/input_graphgps_optimization/O12-10-seeds-prediction-models/predict_ensemble_10seed}"

"$PYTHON_BIN" scripts/diagnostics/predict_o12_10seed_ensemble.py \
    --model-root "$MODEL_ROOT" \
    --input-files \
        datasets_lrx/raw/predict/20260723-DOPE-peptide-predict2.csv \
        datasets_lrx/raw/predict/20260723-library-single-predict.csv \
        datasets_lrx/raw/predict/20260723-validation.xlsx \
    --output-root "$OUTPUT_ROOT" \
    --target-group core4

"$PYTHON_BIN" scripts/diagnostics/predict_o12_10seed_ensemble.py \
    --model-root "$MODEL_ROOT" \
    --input-files \
        datasets_lrx/raw/predict/20260723-DOPE-peptide-predict2.csv \
        datasets_lrx/raw/predict/20260723-library-single-predict.csv \
        datasets_lrx/raw/predict/20260723-validation.xlsx \
    --output-root "$OUTPUT_ROOT" \
    --target-group norm2

"$PYTHON_BIN" scripts/diagnostics/merge_o12_10seed_prediction_groups.py \
    --output-root "$OUTPUT_ROOT"

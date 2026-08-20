  PY=/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python
  MANIFESTS=results/input_graphgps_optimization/five_split_manifests
  BASE=results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/source_config.yaml
  MORDRED=results/input_graphgps_optimization/features/mordred11_train_standardized.csv

  for SPLIT_SEED in 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135; do
    $PY scripts/diagnostics/run_fusion_head_experiment.py \
      --config "$BASE" \
      --run-dir "results/input_graphgps_optimization/five_split_runs/O12_split${SPLIT_SEED}" \
      --split-manifest "$MANIFESTS/split_manifest_seed${SPLIT_SEED}.csv" \
      --fold "split${SPLIT_SEED}" --group B --candidate "O12S${SPLIT_SEED}" \
      --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
      --seed 43 --base-lr 0.001 --weight-decay 1e-5 \
      --gt-dropout 0.1 --gt-attn-dropout 0.2 \
      --use-mordred-features --mordred-feature-dim 11 \
      --mordred-feature-path "$MORDRED" \
      --use-component-aux-features \
      --execution-max-epochs 300 --include-test
  done

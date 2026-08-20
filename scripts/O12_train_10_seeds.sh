  PY=/home/puzexuan/anaconda3/envs/biology_prediction_gpu/bin/python
  MANIFESTS=results/input_graphgps_optimization/five_split_manifests
  BASE=results/input_graphgps_optimization/experiments/O12_input_onehot_aux_all_mordred_attn20_seed43/source_config.yaml

  for SPLIT_SEED in 100 101 102 103 104 105 106 107 108 109; do
    $PY scripts/diagnostics/run_fusion_head_experiment.py \
      --config "$BASE" \
      --run-dir "results/input_graphgps_optimization/five_split_runs/O12_split${SPLIT_SEED}" \
      --split-manifest "$MANIFESTS/split_manifest_seed${SPLIT_SEED}.csv" \
      --fold "split${SPLIT_SEED}" --group B --candidate "O12S${SPLIT_SEED}" \
      --fusion-type concat_mlp --head-type baseline --model-type OneHotEmbedGPS \
      --seed 43 --base-lr 0.001 --weight-decay 1e-5 \
      --gt-dropout 0.1 --gt-attn-dropout 0.2 \
      --execution-max-epochs 300 --include-test \
      --enable-norm-sigmoid-weighting --norm-weight-low 0.1 --norm-weight-high 20\
  done

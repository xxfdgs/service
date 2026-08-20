python python/analyze_gnn_embeddings.py \
    --config results/coarse_grain_noaux/direct_train_coarse_noaux/config.yaml \
    --checkpoint results/coarse_grain_noaux/direct_train_coarse_noaux/0/ckpt/42.ckpt \
    --csv datasets_lrx/raw/input/20260703_sum.csv \
    --dataset-name train \
    --output-dir analysis/oarse_grain_noaux_5 \
    --component 5
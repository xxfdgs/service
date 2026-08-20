python python/analyze_gnn_embeddings.py \
    --config results/fifth_component_weight2_aux/direct_train_aux/config.yaml \
    --checkpoint results/fifth_component_weight2_aux/direct_train_aux/0/ckpt/66.ckpt \
    --csv datasets_lrx/raw/feedback/20260703_validation.csv \
    --dataset-name feedback \
    --output-dir analysis/fifth_component_aux_seed0_feedback_5 \
    --component 5
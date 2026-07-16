python main.py --cfg configs/GPS/direct_train_coarse_noaux.yaml --repeat 10 \
  out_dir results/coarse_mordred \
  use_mordred_features True \
  mordred_feature_dim 11 \
  mordred_feature_path results/mordred_train_feedback/mordred_selected_features.csv
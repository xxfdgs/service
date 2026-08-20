Simple molecular baselines — real-data v2

Verified here:
- Real uploaded training data: 700 rows
- Fifth-identity OOD seeds 100, 101, 102
- Targets: Norm_before, Norm_after
- 8 default baselines
- Plotting code syntax checked

new_validation plotting:
For every target/model, the script creates:
  new_validation_scatter_plots/scatter_<target>_<model>.png
  new_validation_scatter_plots/scatter_<target>_<model>.pdf

Each figure has two separate panels:
  - single
  - double

Each panel includes:
  - y = x
  - x = 1
  - y = 1
  - MAE, R2, Spearman, Recall>1, F2>1, FN, FP

The ChatGPT runtime did not have the original 26-row new_validation.csv mounted,
so external inference itself could not be executed here. On the project server,
the default NEW_VALIDATION path is datasets_lrx/raw/feedback/new_validation.csv.

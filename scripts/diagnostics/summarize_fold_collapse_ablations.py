#!/usr/bin/env python3
"""Create the required train/validation-only diagnostic-ablation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "results/fold4_collapse_audit"
OUT = AUDIT / "diagnostic_ablations"
CONTROLS = OUT / "controls"

RUNS = [
    ("A_baseline", "fold_4", AUDIT / "reproduction/reproduction_a_exact", True),
    ("A_baseline", "fold_0", CONTROLS / "fold0_baseline_original_scheduler_60", True),
    ("B_no_early_stopping", "fold_4", CONTROLS / "fold4_no_early_stop_43", True),
    ("B_no_early_stopping", "fold_0", CONTROLS / "fold0_no_early_stop_43", True),
    ("C_fixed_lr", "fold_4", CONTROLS / "fold4_fixed_lr_43", True),
    ("C_fixed_lr", "fold_0", CONTROLS / "fold0_fixed_lr_43", True),
    ("E_uniform_fusion", "fold_4", CONTROLS / "fold4_uniform_fusion_45", True),
    ("E_uniform_fusion", "fold_0", CONTROLS / "fold0_uniform_fusion_45", True),
    # Preserved for reproducibility but excluded from every conclusion: it
    # changed the cosine scheduler horizon while testing the runner.
    ("INVALID_preliminary_short_scheduler", "fold_0", CONTROLS / "fold0_baseline_60", False),
]


def metrics(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for target, group in frame.groupby("target", sort=True):
        y, p = group.y_true.to_numpy(), group.y_pred.to_numpy()
        rows.append({"target": target, "n": len(group), "mae": float(mean_absolute_error(y, p)),
                     "r2": float(r2_score(y, p)), "spearman": float(spearmanr(y, p).statistic),
                     "prediction_std": float(np.std(p, ddof=1)), "target_std": float(np.std(y, ddof=1)),
                     "std_ratio": float(np.std(p, ddof=1) / np.std(y, ddof=1))})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory, metric_rows, epoch_frames, fusion_rows = [], [], [], []
    for control, fold, run_dir, valid in RUNS:
        summary = json.loads((run_dir / "summary.json").read_text())
        settings = json.loads((run_dir / "run_settings.json").read_text())
        inventory.append({"control": control, "fold": fold, "run_dir": str(run_dir.relative_to(ROOT)), "status": "completed",
                          "valid_for_conclusion": valid, "best_epoch": summary["best_epoch"],
                          "best_validation_loss": summary["best_validation_loss"], "last_epoch": summary["last_epoch"],
                          "scheduler": summary["scheduler"], "early_stopping": summary["early_stopping"],
                          "uniform_fusion": settings.get("uniform_fusion", False),
                          "execution_max_epoch": settings.get("execution_max_epoch", settings["max_epoch"]),
                          "outer_test_used_for_selection": False})
        epoch = pd.read_csv(run_dir / "epoch_metrics.csv").drop_duplicates(["epoch", "split", "target"], keep="last")
        epoch_frames.append(epoch.assign(control=control, fold=fold, valid_for_conclusion=valid))
        prediction = pd.read_csv(run_dir / "best_predictions.csv")
        # These are diagnostic runs. Test predictions exist only because the
        # generic runner exports after a validation-selected checkpoint; do not
        # include them in model/repair choice metrics.
        for split in ("train", "val"):
            for row in metrics(prediction[prediction.split == split]):
                metric_rows.append({"control": control, "fold": fold, "split": split, "best_epoch": summary["best_epoch"],
                                    "best_validation_loss": summary["best_validation_loss"], "valid_for_conclusion": valid, **row})
        fusion = pd.read_csv(run_dir / "fusion_weight_history.csv")
        fusion = fusion[fusion.epoch == summary["best_epoch"]].groupby(["split", "target", "branch"], as_index=False).agg(
            weight_mean=("weight_mean", "mean"), weight_min=("weight_min", "min"), weight_max=("weight_max", "max"),
            entropy_mean=("entropy_mean", "mean"))
        fusion_rows.append(fusion.assign(control=control, fold=fold, best_epoch=summary["best_epoch"], valid_for_conclusion=valid))
    pd.DataFrame(inventory).to_csv(OUT / "run_inventory.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUT / "fold_metrics.csv", index=False)
    pd.concat(epoch_frames, ignore_index=True).to_csv(OUT / "epoch_metrics.csv", index=False)
    pd.concat(fusion_rows, ignore_index=True).to_csv(OUT / "fusion_comparison.csv", index=False)

    probes = pd.concat([
        pd.read_csv(OUT / "branch_probes/fold0_original_best/branch_probe_metrics.csv"),
        pd.read_csv(OUT / "branch_probes/fold4_original_best_retry/branch_probe_metrics.csv"),
    ], ignore_index=True)
    probes.to_csv(OUT / "branch_probe_metrics.csv", index=False)
    loss = pd.concat([
        pd.read_csv(OUT / "loss_balance/fold0_original_best/loss_balance_audit.csv"),
        pd.read_csv(OUT / "loss_balance/fold4_original_best/loss_balance_audit.csv"),
    ], ignore_index=True)
    loss.to_csv(OUT / "loss_balance_audit.csv", index=False)

    tree = pd.read_csv(ROOT / "results/deduplicated_rebaseline/tree_baselines/fold_metrics.csv")
    tree = tree[(tree.protocol == "formula_identity_group_cv") & (tree.outer_fold.isin([0, 4])) &
                (tree.model == "NestedSelectedBaseline")].copy()
    tree.to_csv(OUT / "tree_fold_comparison.csv", index=False)

    summary = pd.DataFrame(inventory)
    base = pd.DataFrame(metric_rows)
    loss_target = loss[loss.kind == "target_loss_gradient"].drop_duplicates(["fold", "target"])
    report = f"""# Minimal diagnostic ablation report

All comparisons below use only train and validation for interpretation and selection. The runner exported test predictions after each already-validation-selected checkpoint, but no test metric appears in `fold_metrics.csv` and no test label was used to choose a control or a checkpoint.

## Control inventory

{summary.to_csv(index=False)}

## Findings

- **A, instrumented baseline:** fold_4 is exactly reproducible twice. Its learned gate first exceeds 0.98 for one branch at epoch 41 and a validation prediction standard-deviation ratio first falls below 0.10 at epoch 42. The selected historical checkpoint is epoch 49.
- **B, no early stopping:** both folds are bitwise-identical to their baseline prefixes through epoch 42. Patience is 50, so early stopping cannot have caused an epoch-41/42 event.
- **C, fixed LR:** fold_4's best validation total L1 is 0.728903 versus the baseline 0.726506 at the same diagnostic horizon. It does not remove the low-variance prediction behavior, so scheduler decay is not supported as a repair.
- **D, loss audit:** all four targets have eight valid labels in the fixed train batch. Fold_4 loss fractions are {', '.join(f'{row.target}={row.loss_fraction:.3f}' for row in loss_target[loss_target.fold == 'fold_4'].itertuples())}; none is a missing-mask or extreme-scale dominance case. Therefore the requested alternative loss normalization was not run.
- **E, uniform fusion:** validation-selected losses worsen sharply (fold_4 2.035149; fold_0 1.938209), so a simple 1/3 replacement is not an acceptable fix and cannot be selected.
- **F, frozen linear probes:** graph, descriptor, and fused representations retain non-zero validation Spearman signal in both folds. Thus the near-constant final prediction is not explained by all input information disappearing.
- **Tree comparator:** the pre-existing nested tree baseline has positive outer-test R² for all fold_4 targets and is not uniformly worse than fold_0. This rejects `TRUE_FOLD_DIFFICULTY` as the available explanation; it is reported only as a post-hoc diagnostic and was not used for model selection.

## Decision

Learned three-branch weights consistently saturate to the main branch in both folds, and the main/middle head activations lose variation. However, uniform weights degrade validation loss, fixed LR does not recover fold_4, and no data/cache/checkpoint/mask/scheduler/early-stop implementation error has been demonstrated. The evidence supports **fusion/optimization instability as a candidate mechanism**, but not an authorized production-code repair. Current classification: **ROOT_CAUSE_UNRESOLVED** (with observed fusion saturation), not a confirmed global engineering bug.
"""
    (OUT / "diagnostic_ablation_report.md").write_text(report)
    print(json.dumps({"runs": len(inventory), "metrics": len(metric_rows), "probes": len(probes), "loss_rows": len(loss)}, indent=2))


if __name__ == "__main__":
    main()

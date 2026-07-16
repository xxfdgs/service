#!/usr/bin/env python3
"""Write an evidence-linked final report when no production fix is authorized."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "results/fold4_collapse_audit"


def macro_checkpoint_metrics() -> dict[str, dict[str, float]]:
    data = pd.read_csv(AUDIT / "checkpoint_audit/checkpoint_prediction_metrics.csv")
    return {split: group[["mae", "std_ratio", "r2", "spearman"]].mean().to_dict()
            for split, group in data[data.fold == "fold_4"].groupby("split")}


def macro_runner_metrics(path: Path) -> dict[str, dict[str, float]]:
    data = pd.read_csv(path / "best_predictions.csv")
    rows = []
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score
    for split, split_frame in data.groupby("split"):
        values = []
        for _, frame in split_frame.groupby("target"):
            values.append({"mae": mean_absolute_error(frame.y_true, frame.y_pred), "std_ratio": frame.y_pred.std(ddof=1) / frame.y_true.std(ddof=1),
                           "r2": r2_score(frame.y_true, frame.y_pred), "spearman": spearmanr(frame.y_true, frame.y_pred).statistic})
        rows.append((split, pd.DataFrame(values).mean(numeric_only=True).to_dict()))
    return dict(rows)


def main() -> None:
    checkpoint = macro_checkpoint_metrics()
    f0_path = AUDIT / "diagnostic_ablations/controls/fold0_baseline_original_scheduler_60"
    f0 = macro_runner_metrics(f0_path)
    f4_dyn = AUDIT / "reproduction/reproduction_a_exact"
    f0_summary = pd.read_json(f0_path / "summary.json", typ="series")
    f4_summary = pd.read_json(f4_dyn / "summary.json", typ="series")
    f4_fusion = pd.read_csv(f4_dyn / "fusion_weight_history.csv")
    f0_fusion = pd.read_csv(f0_path / "fusion_weight_history.csv")
    f4_entropy = f4_fusion[(f4_fusion.epoch == int(f4_summary.best_epoch)) & (f4_fusion.split == "val")].entropy_mean.mean()
    f0_entropy = f0_fusion[(f0_fusion.epoch == int(f0_summary.best_epoch)) & (f0_fusion.split == "val")].entropy_mean.mean()
    f4_grad = pd.read_csv(f4_dyn / "gradient_norm_history.csv")
    f0_grad = pd.read_csv(f0_path / "gradient_norm_history.csv")
    f4_head_grad = f4_grad[(f4_grad.epoch == int(f4_summary.best_epoch)) & (f4_grad.module == "main_head")].grad_norm.mean()
    f0_head_grad = f0_grad[(f0_grad.epoch == int(f0_summary.best_epoch)) & (f0_grad.module == "main_head")].grad_norm.mean()

    fix = AUDIT / "fix"
    fixed = AUDIT / "fixed_runs"
    fix.mkdir(exist_ok=True)
    fixed.mkdir(exist_ok=True)
    (fix / "changed_files.txt").write_text(
        "No production repair was authorized.\n\n"
        "Diagnostic-only additions (default behavior preserved):\n"
        "- scripts/diagnostics/run_fold_collapse_reproduction.py\n"
        "- scripts/diagnostics/summarize_fold_collapse_reproduction.py\n"
        "- scripts/diagnostics/run_fold_branch_probes.py\n"
        "- scripts/diagnostics/audit_multitask_loss_balance.py\n"
        "- scripts/diagnostics/summarize_fold_collapse_ablations.py\n"
        "- graphgps/config/config_gps.py (diagnostic_uniform_fusion=False default)\n"
        "- graphgps/network/double_gps_cat_v31_muliti_4_v0.py (opt-in uniform diagnostic branch only)\n")
    (fix / "patch_summary.md").write_text(
        "# No production fix applied\n\n"
        "No data, cache, loss, scheduler, checkpoint, or sample-alignment bug was confirmed. "
        "The only network change is an opt-in diagnostic branch guarded by `cfg.diagnostic_uniform_fusion`; "
        "its default is `False`, retaining the original `torch.softmax` expression. "
        "Uniform fusion degraded inner-validation loss in both folds, so it is not an accepted repair.\n")
    (fix / "regression_tests.md").write_text(
        "# Diagnostic regression checks\n\n"
        "- `py_compile` passed for all added/modified diagnostic scripts and the opt-in model/config code.\n"
        "- Original fold_4 runs A/B have identical canonical epoch metrics and predictions (max difference 0).\n"
        "- Historical 49.ckpt and reproduced selected state differ by at most 5.96e-8.\n"
        "- The default diagnostic switch is false; no historical YAML enables it.\n"
        "- Static audit already asserts sample_id, manifest, feature/cache hashes, label order, finite values, and group isolation.\n")
    placeholder = pd.DataFrame([{"status": "NOT_RUN", "reason": "No confirmed engineering bug; a production fix and its test evaluation would violate the task constraint."}])
    placeholder.to_csv(fixed / "fold_metrics.csv", index=False)
    pd.DataFrame(columns=["sample_id", "split", "target", "y_true", "y_pred"]).to_csv(fixed / "predictions.csv", index=False)
    placeholder.to_csv(fixed / "variance_metrics.csv", index=False)
    placeholder.to_csv(fixed / "dynamics_comparison.csv", index=False)
    (fixed / "fixed_run_report.md").write_text(
        "# Fixed-run status\n\nNo fixed run was executed: an engineering bug was not confirmed, and the specified constraints prohibit treating a diagnostic architecture change as a production fix.\n")

    table = f"""| item | fold_0 | fold_4_before | fold_4_after | conclusion |
| --- | ---: | ---: | ---: | --- |
| train_mae | {f0['train']['mae']:.3f}† | {checkpoint['train']['mae']:.3f} | N/A | no authorized after-run |
| val_mae | {f0['val']['mae']:.3f}† | {checkpoint['val']['mae']:.3f} | N/A | collapse checkpoint is selected by val loss |
| test_mae | {f0['test']['mae']:.3f}† | {checkpoint['test']['mae']:.3f} | N/A | not used for any choice |
| train_std_ratio | {f0['train']['std_ratio']:.3f}† | {checkpoint['train']['std_ratio']:.3f} | N/A | low variation exists in both baseline heads |
| val_std_ratio | {f0['val']['std_ratio']:.3f}† | {checkpoint['val']['std_ratio']:.3f} | N/A | fold_4 is near-constant |
| test_std_ratio | {f0['test']['std_ratio']:.3f}† | {checkpoint['test']['std_ratio']:.3f} | N/A | fold_4 remains near-constant |
| spearman | {f0['val']['spearman']:.3f}† | {checkpoint['val']['spearman']:.3f} | N/A | no ranking recovery demonstrated |
| best_epoch | {int(f0_summary.best_epoch)}† | {int(f4_summary.best_epoch)} | N/A | all selection is validation-only |
| fusion_entropy | {f0_entropy:.3f}† | {f4_entropy:.3f} | N/A | low entropy/main-branch saturation |
| head_gradient_norm | {f0_head_grad:.3f}† | {f4_head_grad:.3f} | N/A | no global NaN/Inf failure |

† fold_0 figures are the valid 60-epoch original-scheduler diagnostic baseline, not a claimed replacement for its historical full 1500-epoch training.
"""
    report = f"""# GraphGPS formula-identity fold 4 collapse audit

## Final state: ROOT_CAUSE_UNRESOLVED

No production fix was applied and no five-fold retraining is authorized. The data/cache/checkpoint paths are validated, and the observed learned-fusion saturation is real; however, the tested minimal diagnostic alternatives do not recover validation quality. Calling any of them a repair would violate the constraint that a fix require confirmed engineering or optimization error.

## Answers to the required questions

1. **Stable reproduction?** Yes. Two independent seed-0 runs are exact at every canonical epoch metric and final sample prediction; both select epoch 49 with validation total L1 0.71741426.
2. **Earliest collapse?** The learned branch weight first exceeds 0.98 at epoch 41; a validation prediction std ratio first falls below 0.10 at epoch 42. Main/middle head activations lose variation while raw feature inputs remain variable.
3. **Main fold difference?** Fold_4's historical best is selected after the gate has saturated and outputs are near constant. Fold_0 also shows gate saturation, but retains relatively more validation ranking signal and a lower selected loss.
4. **Data/manifests/cache?** PASS: no group leakage, sample/cache misalignment, finite-value failure, zero-feature pathology, or manifest error.
5. **Label masks/order?** PASS: four targets have correct `y,y1,y2,y3` ordering, fixed `/100` scale, and valid labels in every audited batch.
6. **49.ckpt correct?** Yes: it is the validation-loss argmin and metadata/hash match the intended fold/config/features.
7. **Scheduler early decay?** No: at collapse it is in warm-up/increasing; at original stop LR is ~0.000997. Fixed LR does not improve fold_4 validation selection.
8. **Early stopping early?** No: collapse is epochs 41–42; patience is 50 and early stop occurs at epoch 100. Disabled-early-stop control is identical through epoch 42.
9. **Gradient failure?** No NaN/Inf. Some inactive/unused paths have zero gradients, but the selected main head has finite gradients; there is no global explosion/vanishing event that explains the collapse.
10. **Multi-task loss imbalance?** No mask/scale error: all targets are valid and fold_4 first-batch L1 fractions are 0.231/0.224/0.317/0.228. The normalized-loss alteration was therefore not run.
11. **First lost variance?** Main and middle prediction-head activation variation collapses before final prediction variance; raw graph/descriptor/formula inputs do not.
12. **Softmax fusion collapse?** Observed: weights saturate to the main branch (about 0.995–0.998 at selected checkpoints) with entropy ~0.02–0.03.
13. **Fused signal lost?** Frozen linear probes retain nonzero validation signal in graph, descriptor, and fused inputs, while final predictions are near constant. This is an output/head/gating-stage pathology, not proof that features contain no signal.
14. **Root class?** `ROOT_CAUSE_UNRESOLVED`: observed fusion/optimization instability is a candidate mechanism, but uniform fusion worsens validation loss, so it is not a confirmed causal repair.
15. **Minimal fix?** None. The diagnostic uniform-fusion switch is default-off and was not promoted to production behavior.
16. **Fold_4 recovery after fix?** Not applicable; no safe fix was demonstrated.
17. **Fold_0 degradation after fix?** Not applicable; no safe fix was demonstrated.
18. **Retrain all folds?** No. `NO_FIX_MODEL_LIMITATION` for retraining scope: no global implementation bug is established.
19. **Proceed to single-task experiments?** Only as a separately registered exploratory protocol, not as a claimed remedy or replacement for this multi-task result.
20. **Next priority?** A validation-only, separately pre-registered fusion redesign study (e.g. residual/concat/gated-concat) with fold_0/fold_4 controls. It is outside this task's authorized minimal-fix scope.

## Quantitative status

{table}

## Evidence and artifacts

- Static audit: `static_audit/static_audit_report.md`
- Checkpoint audit: `checkpoint_audit/checkpoint_selection_report.md`
- Exact reproduction: `reproduction/reproduction_report.md`
- Dynamics: `dynamics/collapse_timeline.csv`
- Diagnostic controls: `diagnostic_ablations/diagnostic_ablation_report.md`
- Execution provenance: `execution_manifest.json`
- No-fix decision: `fix/patch_summary.md`, `fixed_runs/fixed_run_report.md`
"""
    (AUDIT / "report.md").write_text(report)
    print(AUDIT / "report.md")


if __name__ == "__main__":
    main()

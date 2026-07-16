#!/usr/bin/env python3
"""Write provenance for every material command in the fold-collapse audit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "results/fold4_collapse_audit"
CONFIGS = ROOT / "results/deduplicated_rebaseline/graphgps_cv/configs"
DATASET = ROOT / "results/deduplicated_rebaseline/data_audit/dataset_with_sample_id.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(config: Path) -> dict[str, str]:
    payload = yaml.safe_load(config.read_text())
    manifest = ROOT / payload["train"]["manifest_path"]
    feature = ROOT / payload.get("mordred_feature_path", "")
    return {"dataset_hash": sha256(DATASET), "manifest_hash": sha256(manifest),
            "feature_hash": sha256(feature), "config_hash": sha256(config)}


def entry(command: str, run_type: str, fold: str, config: Path, output: Path, checkpoint: str = "", status: str = "completed", error: str = "") -> dict[str, object]:
    item = {"timestamp": datetime.now(timezone.utc).isoformat(), "command": command, "run_type": run_type,
            "fold": fold, "seed": 0, "checkpoint": checkpoint, "status": status, "error": error,
            "output_path": str(output.relative_to(ROOT))}
    item.update(provenance(config))
    return item


def main() -> None:
    f0 = CONFIGS / "formula_identity_group_cv_fold_0_seed_0.yaml"
    f4 = CONFIGS / "formula_identity_group_cv_fold_4_seed_0.yaml"
    ckpt0 = "results/deduplicated_rebaseline/graphgps_cv/training/formula_identity_group_cv_fold_0_seed_0/0/ckpt/106.ckpt"
    ckpt4 = "results/deduplicated_rebaseline/graphgps_cv/training/formula_identity_group_cv_fold_4_seed_0/0/ckpt/49.ckpt"
    records = [
        entry("python scripts/diagnostics/audit_graphgps_fold_collapse.py --output-dir results/fold4_collapse_audit/static_audit", "static_audit", "fold_0,fold_1,fold_4", f4, AUDIT / "static_audit"),
        entry("python scripts/diagnostics/audit_graphgps_checkpoints.py --output-dir results/fold4_collapse_audit/checkpoint_audit", "checkpoint_audit", "fold_0..fold_4", f4, AUDIT / "checkpoint_audit"),
        entry("python scripts/diagnostics/run_fold_collapse_reproduction.py ... --fold fold_4 (resumable A)", "reproduction", "fold_4", f4, AUDIT / "reproduction/reproduction_a_exact", ckpt4),
        entry("python scripts/diagnostics/run_fold_collapse_reproduction.py ... --fold fold_4 (resumable B)", "reproduction", "fold_4", f4, AUDIT / "reproduction/reproduction_b_exact", ckpt4),
    ]
    for control, fold, suffix, extra in [
        ("baseline", "fold_0", "fold0_baseline_original_scheduler_60", "--stop-after-epochs 60"),
        ("no_early_stopping", "fold_0", "fold0_no_early_stop_43", "--stop-after-epochs 43 --early-stopping disabled"),
        ("no_early_stopping", "fold_4", "fold4_no_early_stop_43", "--stop-after-epochs 43 --early-stopping disabled"),
        ("fixed_lr", "fold_0", "fold0_fixed_lr_43", "--stop-after-epochs 43 --scheduler none"),
        ("fixed_lr", "fold_4", "fold4_fixed_lr_43", "--stop-after-epochs 43 --scheduler none"),
        ("uniform_fusion", "fold_0", "fold0_uniform_fusion_45", "--stop-after-epochs 45 --uniform-fusion"),
        ("uniform_fusion", "fold_4", "fold4_uniform_fusion_45", "--stop-after-epochs 45 --uniform-fusion"),
    ]:
        config, ckpt = (f0, ckpt0) if fold == "fold_0" else (f4, ckpt4)
        records.append(entry(f"python scripts/diagnostics/run_fold_collapse_reproduction.py --fold {fold} {extra} (resumable chunks)",
                             f"diagnostic_{control}", fold, config, AUDIT / "diagnostic_ablations/controls" / suffix, ckpt))
    for fold, config, checkpoint, suffix in [("fold_0", f0, ckpt0, "fold0_original_best"), ("fold_4", f4, ckpt4, "fold4_original_best_retry")]:
        records.append(entry("python scripts/diagnostics/run_fold_branch_probes.py ...", "branch_probe", fold, config,
                             AUDIT / "diagnostic_ablations/branch_probes" / suffix, checkpoint))
        records.append(entry("python scripts/diagnostics/audit_multitask_loss_balance.py ...", "loss_balance_audit", fold, config,
                             AUDIT / "diagnostic_ablations/loss_balance" / ("fold0_original_best" if fold == "fold_0" else "fold4_original_best"), checkpoint))
    records.append(entry("python scripts/diagnostics/summarize_fold_collapse_reproduction.py ...", "reproduction_summary", "fold_4", f4, AUDIT / "reproduction", ckpt4))
    records.append(entry("python scripts/diagnostics/summarize_fold_collapse_ablations.py", "ablation_summary", "fold_0,fold_4", f4, AUDIT / "diagnostic_ablations"))
    records.append(entry("python scripts/diagnostics/write_fold_collapse_final_report.py", "final_report", "fold_0,fold_4", f4, AUDIT))
    (AUDIT / "execution_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"records": len(records), "path": str(AUDIT / "execution_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()

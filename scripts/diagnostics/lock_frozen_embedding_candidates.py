#!/usr/bin/env python3
"""Lock one development-qualified epoch-best frozen probe per target for fold 1."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# All selections use the historical validation-best epoch rule so the same
# pre-existing, validation-selected checkpoint rule applies to fold 1.  These
# mappings are chosen solely from fold 0/fold 4 development validation:
# MAE first, with the aerosol descriptor choice preferring the better balanced
# fold-4 R2/Spearman over the small mean-MAE advantage of fused embedding.
LOCK = {
    "EE_before": ("descriptor_branch_raw", "P5_RandomForest"),
    "EE_after": ("descriptor_branch_raw", "P5_RandomForest"),
    "Aerosolization_Efficiency": ("descriptor_branch_raw", "P5_RandomForest"),
    "mRNA_Recovery_Efficiency": ("fused_embedding", "P5_RandomForest"),
}


def main() -> None:
    root = ROOT / "results/frozen_embedding_signal_exp"
    screen = pd.read_csv(root / "probes/candidate_screening.csv")
    rows = []
    for target, (embedding, probe) in LOCK.items():
        hit = screen.loc[(screen.target == target) & (screen.embedding_name == embedding) &
                         (screen.epoch_label == "epoch_best") & (screen.probe == probe) & screen.accepted]
        if len(hit) != 1:
            raise RuntimeError(f"Locked candidate is not uniquely accepted: {target}/{embedding}/{probe}: {len(hit)}")
        rows.append(hit.iloc[0].to_dict())
    output = root / "stage1"
    output.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "candidate_lock.csv", index=False)
    (output / "candidate_lock.json").write_text(json.dumps({
        "epoch_rule": "historical validation-selected epoch_best for each outer fold",
        "probe_type": "P5_RandomForest; identical inner GroupKFold hyperparameter-selection grid retained in fold 1",
        "preprocessing": "none for tree probes; labels/features are never fit on validation/test",
        "candidates": frame.to_dict(orient="records"),
        "selection_scope": "fold_0 and fold_4 explicit validation only; no outer-test used",
    }, indent=2) + "\n")
    print(frame[["target", "embedding_name", "epoch_label", "probe", "mean_val_mae", "mean_val_spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()

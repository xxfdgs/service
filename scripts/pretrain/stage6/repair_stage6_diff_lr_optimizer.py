#!/usr/bin/env python3
"""Repair the Stage-6 differential-LR optimizer construction.

Problem
-------
PyG GraphGym create_optimizer() assumes an iterable of Parameter objects and
internally executes:

    filter(lambda p: p.requires_grad, params)

Therefore standard PyTorch parameter-group dictionaries are rejected with:

    AttributeError: 'dict' object has no attribute 'requires_grad'

Fix
---
- normal single-LR path: keep GraphGym create_optimizer() unchanged;
- differential-LR path: construct torch.optim.AdamW directly from the two
  validated parameter-group dictionaries.

The project's registered AdamW optimizer currently delegates to torch AdamW
with lr and weight_decay only, so this preserves the optimizer implementation
and defaults while enabling native PyTorch per-group learning rates.

The patch is fail-closed and idempotent.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


OLD = """    optimizer = create_optimizer(
        optimizer_parameters,
        optimizer_config,
    )

    optimizer_group_metadata['optimizer_group_lrs_before_scheduler'] = [
"""

NEW = """    if args.comp5_lr is None:
        # Preserve the historical GraphGym optimizer path exactly for all
        # non-differential-LR experiments.
        optimizer = create_optimizer(
            optimizer_parameters,
            optimizer_config,
        )
    else:
        # GraphGym create_optimizer() filters its input as if every element is
        # a Parameter object and therefore cannot accept PyTorch parameter-
        # group dictionaries.  The registered optimizer used by this project
        # is AdamW, so use native torch.optim.AdamW for the differential-LR
        # branch only.
        optimizer_name = str(cfg.optim.optimizer).strip().lower().replace('_', '')
        if optimizer_name != 'adamw':
            raise ValueError(
                '--comp5-lr currently supports only the project AdamW optimizer; '
                f'effective cfg.optim.optimizer={cfg.optim.optimizer!r}.'
            )

        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=float(cfg.optim.base_lr),
            weight_decay=float(cfg.optim.weight_decay),
        )

    optimizer_group_metadata['optimizer_group_lrs_before_scheduler'] = [
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/diagnostics/run_fusion_head_experiment.py"),
    )
    parser.add_argument(
        "--backup-suffix",
        default=".pre_stage6_optimizer_repair_backup",
    )
    args = parser.parse_args()

    runner = args.runner.resolve()
    if not runner.is_file():
        raise FileNotFoundError(runner)

    text = runner.read_text(encoding="utf-8")

    repaired_marker = "GraphGym create_optimizer() filters its input"
    if repaired_marker in text:
        compile(text, str(runner), "exec")
        print(f"Stage-6 optimizer repair already present: {runner}")
        print("Syntax check: PASS")
        return

    required = [
        "--comp5-lr",
        "optimizer_parameters = [",
        "optimizer_parameter_groups.json",
        "Differential LR audit PASS",
    ]
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(
            "The expected Stage-6 differential-LR patch is not fully present; "
            "missing markers: " + ", ".join(missing)
        )

    count = text.count(OLD)
    if count != 1:
        raise RuntimeError(
            f"Expected optimizer-construction anchor exactly once, found {count}. "
            "Refusing blind modification."
        )

    patched = text.replace(OLD, NEW, 1)
    compile(patched, str(runner), "exec")

    backup = runner.with_name(runner.name + args.backup_suffix)
    if backup.exists():
        raise FileExistsError(
            f"Repair backup already exists: {backup}. "
            "Inspect before modifying again."
        )

    shutil.copy2(runner, backup)
    runner.write_text(patched, encoding="utf-8")
    compile(runner.read_text(encoding="utf-8"), str(runner), "exec")

    print(f"Repaired: {runner}")
    print(f"Backup  : {backup}")
    print("Syntax check: PASS")
    print("Single-LR path: GraphGym create_optimizer()")
    print("Diff-LR path  : torch.optim.AdamW(parameter_groups)")


if __name__ == "__main__":
    main()

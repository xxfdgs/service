#!/usr/bin/env python3
"""
Patch the current downstream runner with an auditable Stage-4 Comp5 transfer hook.

Target:
    scripts/diagnostics/run_fusion_head_experiment.py

What is added
-------------
CLI:
    --comp5-pretrained-checkpoint PATH
    --comp5-pretrain-label LABEL

Fresh-run order:
    create_model_gps()
        -> strict load into model.model.comp5_encoder
        -> create_optimizer()

The GraphGym wrapper returned by create_model_gps() stores the actual
OneHotEmbedGPSModel in `.model`, so Stage-4 transfer must target model.model.

For every run a comp5_initialization.json is written:
    P0: random initialization
    P1/P2: checkpoint path, SHA256 and strict load report

Checkpoint metadata also receives the same initialization provenance.

The patch is fail-closed and idempotent.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMPORT_ANCHOR = (
    "from graphgps.create_model_gps import create_model_gps  # noqa: E402\n"
)

IMPORT_INSERT = (
    "from graphgps.create_model_gps import create_model_gps  # noqa: E402\n"
    "from scripts.pretrain.stage4.stage4_transfer import (  # noqa: E402\n"
    "    load_stage4_comp5_encoder,\n"
    ")\n"
)

CLI_ANCHOR = (
    "    parser.add_argument('--resume', action='store_true')\n"
)

CLI_INSERT = """    parser.add_argument(
        '--comp5-pretrained-checkpoint',
        type=Path,
        default=None,
        help=(
            'Optional Stage-4 Comp5GraphEncoder checkpoint. '
            'Loaded strictly into OneHotEmbedGPSModel.comp5_encoder '
            'after model construction and before optimizer construction.'
        ),
    )
    parser.add_argument(
        '--comp5-pretrain-label',
        type=str,
        default='P0_random',
        help='Audit label for Fifth-encoder initialization, e.g. P0_random, P1_PT_D, P2_PT_DF.',
    )
    parser.add_argument('--resume', action='store_true')
"""

MODEL_ANCHOR = """    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model = create_model_gps().to(device)
    optimizer = create_optimizer(model.parameters(), OptimizerConfig(
"""

MODEL_INSERT = """    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model = create_model_gps().to(device)

    # ------------------------------------------------------------------
    # Stage-5 Fifth-encoder initialization.
    #
    # create_model_gps() returns GraphGymModule, whose `.model` is the
    # registered OneHotEmbedGPSModel.  Load BEFORE optimizer construction so
    # optimizer state starts from the transferred parameters.
    # ------------------------------------------------------------------
    comp5_init_metadata = {
        'label': str(args.comp5_pretrain_label),
        'mode': 'random',
        'checkpoint': None,
        'checkpoint_sha256': None,
        'strict_transfer_report': None,
    }

    if args.comp5_pretrained_checkpoint is not None:
        checkpoint_path = args.comp5_pretrained_checkpoint.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f'Comp5 pretrained checkpoint is missing: {checkpoint_path}'
            )
        if str(cfg.model.type) != 'OneHotEmbedGPS':
            raise ValueError(
                '--comp5-pretrained-checkpoint requires --model-type OneHotEmbedGPS '
                f'(effective cfg.model.type={cfg.model.type!r}).'
            )
        if not hasattr(model, 'model'):
            raise AttributeError(
                'create_model_gps() result has no `.model` GraphGym wrapper target.'
            )
        transfer_report = load_stage4_comp5_encoder(
            model.model,
            checkpoint_path,
            map_location='cpu',
        )
        comp5_init_metadata.update({
            'mode': 'stage4_pretrained_full_finetune',
            'checkpoint': str(checkpoint_path),
            'checkpoint_sha256': file_sha256(checkpoint_path),
            'strict_transfer_report': transfer_report,
        })

    (run_dir / 'comp5_initialization.json').write_text(
        json.dumps(comp5_init_metadata, indent=2) + '\\n'
    )

    print(
        '[Stage5] Comp5 initialization: '
        f"{comp5_init_metadata['label']} "
        f"mode={comp5_init_metadata['mode']}"
    )
    if comp5_init_metadata['checkpoint'] is not None:
        print(
            '[Stage5] Strict Comp5 transfer PASS: '
            f"{comp5_init_metadata['checkpoint']}"
        )

    optimizer = create_optimizer(model.parameters(), OptimizerConfig(
"""

META_ANCHOR = (
    "    start_epoch, best_loss, best_epoch, best_state = 0, math.inf, None, None\n"
)

META_INSERT = """    checkpoint_metadata['comp5_initialization'] = comp5_init_metadata
    start_epoch, best_loss, best_epoch, best_state = 0, math.inf, None, None
"""


def replace_once(text: str, anchor: str, replacement: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f'Patch anchor {label!r} expected exactly once, found {count}. '
            'Refusing to modify the runner.'
        )
    return text.replace(anchor, replacement, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/diagnostics/run_fusion_head_experiment.py"),
    )
    parser.add_argument(
        "--backup-suffix",
        default=".pre_stage5_backup",
    )
    args = parser.parse_args()

    runner = args.runner.resolve()
    if not runner.is_file():
        raise FileNotFoundError(runner)

    text = runner.read_text(encoding="utf-8")

    marker = "--comp5-pretrained-checkpoint"
    if marker in text and "comp5_initialization.json" in text:
        compile(text, str(runner), "exec")
        print(f"Stage-5 transfer patch already present: {runner}")
        print("Syntax check: PASS")
        return

    backup = runner.with_name(runner.name + args.backup_suffix)
    if backup.exists():
        raise FileExistsError(
            f"Backup already exists but patch marker is absent: {backup}. "
            "Inspect manually before patching."
        )

    patched = text
    patched = replace_once(
        patched, IMPORT_ANCHOR, IMPORT_INSERT, "transfer import"
    )
    patched = replace_once(
        patched, CLI_ANCHOR, CLI_INSERT, "CLI arguments"
    )
    patched = replace_once(
        patched, MODEL_ANCHOR, MODEL_INSERT, "model-before-optimizer"
    )
    patched = replace_once(
        patched, META_ANCHOR, META_INSERT, "checkpoint metadata"
    )

    compile(patched, str(runner), "exec")

    shutil.copy2(runner, backup)
    runner.write_text(patched, encoding="utf-8")

    verify = runner.read_text(encoding="utf-8")
    compile(verify, str(runner), "exec")

    print(f"Patched: {runner}")
    print(f"Backup : {backup}")
    print("Syntax check: PASS")
    print("Stage-5 hook order: create model -> strict Comp5 load -> optimizer")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Patch the Stage-5 downstream runner with differential Comp5 learning rate.

Prerequisite
------------
The runner must already contain the Stage-5 transfer hook:
    --comp5-pretrained-checkpoint
    --comp5-pretrain-label
    load_stage4_comp5_encoder(...)

Adds
----
    --comp5-lr FLOAT

When supplied:
    rest of model      -> cfg.optim.base_lr
    model.model.comp5_encoder -> --comp5-lr

The patch is fail-closed:
- requires a Stage-4 pretrained checkpoint when --comp5-lr is used;
- verifies the two parameter groups form an exact, non-overlapping partition
  of all trainable model parameters;
- verifies optimizer group LRs;
- verifies scheduler base-LR ratio is preserved;
- writes optimizer_parameter_groups.json;
- preserves resume safety for differential-LR runs.

Fresh-run order remains:
    create model
      -> strict Stage-4 Comp5 load
      -> construct two optimizer parameter groups
      -> create optimizer
      -> create scheduler
      -> audit LR ratio
      -> train
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CLI_ANCHOR = "    parser.add_argument('--resume', action='store_true')\n"

CLI_INSERT = """    parser.add_argument(
        '--comp5-lr',
        type=float,
        default=None,
        help=(
            'Optional differential learning rate for model.model.comp5_encoder. '
            'All remaining trainable parameters keep cfg.optim.base_lr. '
            'Requires --comp5-pretrained-checkpoint.'
        ),
    )
    parser.add_argument('--resume', action='store_true')
"""

PROVENANCE_ANCHOR = """    (run_dir / 'comp5_initialization.json').write_text(
        json.dumps(comp5_init_metadata, indent=2) + '\\n'
    )
"""

PROVENANCE_INSERT = """    if args.comp5_lr is not None:
        if args.comp5_lr <= 0:
            raise ValueError('--comp5-lr must be positive.')
        if args.comp5_pretrained_checkpoint is None:
            raise ValueError(
                '--comp5-lr is a Stage-6 pretrained-encoder option and requires '
                '--comp5-pretrained-checkpoint.'
            )

    comp5_init_metadata['differential_lr'] = bool(args.comp5_lr is not None)
    comp5_init_metadata['rest_lr'] = float(cfg.optim.base_lr)
    comp5_init_metadata['comp5_lr'] = (
        float(args.comp5_lr)
        if args.comp5_lr is not None
        else float(cfg.optim.base_lr)
    )

    (run_dir / 'comp5_initialization.json').write_text(
        json.dumps(comp5_init_metadata, indent=2) + '\\n'
    )
"""

OPTIMIZER_ANCHOR = """    optimizer = create_optimizer(model.parameters(), OptimizerConfig(
        optimizer=cfg.optim.optimizer, base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay, momentum=cfg.optim.momentum))
"""

OPTIMIZER_INSERT = """    optimizer_config = OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum,
    )

    optimizer_group_metadata = {
        'differential_lr': bool(args.comp5_lr is not None),
        'rest_lr': float(cfg.optim.base_lr),
        'comp5_lr': (
            float(args.comp5_lr)
            if args.comp5_lr is not None
            else float(cfg.optim.base_lr)
        ),
        'rest_trainable_parameter_count': None,
        'comp5_trainable_parameter_count': None,
        'total_trainable_parameter_count': int(
            sum(parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad)
        ),
        'optimizer_group_lrs_before_scheduler': None,
        'scheduler_base_lrs': None,
        'optimizer_group_lrs_after_scheduler_init': None,
    }

    if args.comp5_lr is None:
        optimizer_parameters = model.parameters()
    else:
        if not hasattr(model, 'model') or not hasattr(model.model, 'comp5_encoder'):
            raise AttributeError(
                '--comp5-lr requires model.model.comp5_encoder, but the current '
                'GraphGym model does not expose that interface.'
            )

        comp5_parameters = [
            parameter
            for parameter in model.model.comp5_encoder.parameters()
            if parameter.requires_grad
        ]
        comp5_ids = {id(parameter) for parameter in comp5_parameters}

        rest_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in comp5_ids
        ]
        rest_ids = {id(parameter) for parameter in rest_parameters}
        all_trainable = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        all_trainable_ids = {id(parameter) for parameter in all_trainable}

        if not comp5_parameters:
            raise RuntimeError(
                'Differential LR requested but Comp5GraphEncoder has no '
                'trainable parameters.'
            )
        if not rest_parameters:
            raise RuntimeError(
                'Differential LR requested but no non-Comp5 trainable '
                'parameters were found.'
            )
        if comp5_ids & rest_ids:
            raise RuntimeError(
                'Comp5/rest optimizer parameter groups overlap.'
            )
        if (comp5_ids | rest_ids) != all_trainable_ids:
            raise RuntimeError(
                'Comp5/rest optimizer parameter groups do not exactly partition '
                'all trainable model parameters.'
            )
        if len(comp5_ids) != len(comp5_parameters):
            raise RuntimeError(
                'Duplicate Comp5 parameter objects detected.'
            )
        if len(rest_ids) != len(rest_parameters):
            raise RuntimeError(
                'Duplicate rest-model parameter objects detected.'
            )

        optimizer_group_metadata.update({
            'rest_trainable_parameter_count': int(
                sum(parameter.numel() for parameter in rest_parameters)
            ),
            'comp5_trainable_parameter_count': int(
                sum(parameter.numel() for parameter in comp5_parameters)
            ),
        })

        if (
            optimizer_group_metadata['rest_trainable_parameter_count']
            + optimizer_group_metadata['comp5_trainable_parameter_count']
            != optimizer_group_metadata['total_trainable_parameter_count']
        ):
            raise RuntimeError(
                'Optimizer parameter-count partition invariant failed.'
            )

        # Put the normal/base-LR group first so existing runner logging
        # scheduler.get_last_lr()[0] continues to report the rest-model LR.
        optimizer_parameters = [
            {
                'params': rest_parameters,
                'lr': float(cfg.optim.base_lr),
            },
            {
                'params': comp5_parameters,
                'lr': float(args.comp5_lr),
            },
        ]

    optimizer = create_optimizer(
        optimizer_parameters,
        optimizer_config,
    )

    optimizer_group_metadata['optimizer_group_lrs_before_scheduler'] = [
        float(group['lr']) for group in optimizer.param_groups
    ]

    if args.comp5_lr is not None:
        actual = optimizer_group_metadata['optimizer_group_lrs_before_scheduler']
        expected = [float(cfg.optim.base_lr), float(args.comp5_lr)]
        if len(actual) != 2 or any(
            abs(got - want) > max(1e-12, abs(want) * 1e-9)
            for got, want in zip(actual, expected)
        ):
            raise RuntimeError(
                'create_optimizer did not preserve differential parameter-group '
                f'learning rates: expected={expected}, actual={actual}'
            )
"""

SCHEDULER_AUDIT_ANCHOR = """    # Prediction rows carry loader source indices.  They must be mapped through
"""

SCHEDULER_AUDIT_INSERT = """    optimizer_group_metadata['optimizer_group_lrs_after_scheduler_init'] = [
        float(group['lr']) for group in optimizer.param_groups
    ]
    scheduler_base_lrs = getattr(scheduler, 'base_lrs', None)
    if scheduler_base_lrs is not None:
        optimizer_group_metadata['scheduler_base_lrs'] = [
            float(value) for value in scheduler_base_lrs
        ]

    optimizer_group_metadata_path = run_dir / 'optimizer_parameter_groups.json'

    if args.comp5_lr is not None:
        expected_ratio = float(args.comp5_lr) / float(cfg.optim.base_lr)

        # Prefer scheduler base_lrs for the hard audit.  Fall back to current
        # optimizer LRs when the scheduler implementation exposes no base_lrs.
        ratio_source = (
            optimizer_group_metadata['scheduler_base_lrs']
            if optimizer_group_metadata['scheduler_base_lrs'] is not None
            else optimizer_group_metadata['optimizer_group_lrs_after_scheduler_init']
        )
        if len(ratio_source) != 2 or ratio_source[0] == 0:
            raise RuntimeError(
                'Cannot audit differential LR ratio after scheduler creation: '
                f'{ratio_source}'
            )
        actual_ratio = float(ratio_source[1]) / float(ratio_source[0])
        if abs(actual_ratio - expected_ratio) > max(
            1e-12, abs(expected_ratio) * 1e-7
        ):
            raise RuntimeError(
                'Scheduler did not preserve the requested Comp5/rest LR ratio: '
                f'expected={expected_ratio}, actual={actual_ratio}, '
                f'LRs={ratio_source}'
            )

        if args.resume:
            if not optimizer_group_metadata_path.is_file():
                raise FileNotFoundError(
                    'Differential-LR resume requested but '
                    f'{optimizer_group_metadata_path} is missing.'
                )
            saved_group_metadata = json.loads(
                optimizer_group_metadata_path.read_text(encoding='utf-8')
            )
            for field in ('rest_lr', 'comp5_lr'):
                requested = float(optimizer_group_metadata[field])
                saved = float(saved_group_metadata[field])
                if abs(requested - saved) > max(1e-12, abs(saved) * 1e-9):
                    raise RuntimeError(
                        'Differential-LR resume provenance mismatch for '
                        f'{field}: saved={saved}, requested={requested}'
                    )
        else:
            optimizer_group_metadata_path.write_text(
                json.dumps(optimizer_group_metadata, indent=2) + '\\n',
                encoding='utf-8',
            )

        print(
            '[Stage6] Differential LR audit PASS: '
            f"rest={optimizer_group_metadata['rest_lr']:.3e}, "
            f"comp5={optimizer_group_metadata['comp5_lr']:.3e}, "
            f"ratio={expected_ratio:.6f}, "
            f"rest_params={optimizer_group_metadata['rest_trainable_parameter_count']}, "
            f"comp5_params={optimizer_group_metadata['comp5_trainable_parameter_count']}"
        )
    elif not args.resume:
        optimizer_group_metadata_path.write_text(
            json.dumps(optimizer_group_metadata, indent=2) + '\\n',
            encoding='utf-8',
        )

    # Prediction rows carry loader source indices.  They must be mapped through
"""


def replace_once(text: str, anchor: str, replacement: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"Stage-6 patch anchor {label!r} expected exactly once, found {count}. "
            "Refusing to modify the runner."
        )
    return text.replace(anchor, replacement, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--runner',
        type=Path,
        default=Path('scripts/diagnostics/run_fusion_head_experiment.py'),
    )
    parser.add_argument(
        '--backup-suffix',
        default='.pre_stage6_difflr_backup',
    )
    args = parser.parse_args()

    runner = args.runner.resolve()
    if not runner.is_file():
        raise FileNotFoundError(runner)

    text = runner.read_text(encoding='utf-8')

    required_stage5 = (
        '--comp5-pretrained-checkpoint',
        '--comp5-pretrain-label',
        'load_stage4_comp5_encoder',
        'comp5_initialization.json',
    )
    missing = [marker for marker in required_stage5 if marker not in text]
    if missing:
        raise RuntimeError(
            'Stage-5 transfer hook is not fully installed; missing: '
            + ', '.join(missing)
        )

    if '--comp5-lr' in text and 'optimizer_parameter_groups.json' in text:
        compile(text, str(runner), 'exec')
        print(f'Stage-6 differential-LR patch already present: {runner}')
        print('Syntax check: PASS')
        return

    backup = runner.with_name(runner.name + args.backup_suffix)
    if backup.exists():
        raise FileExistsError(
            f'Backup already exists while Stage-6 markers are absent: {backup}. '
            'Inspect the runner manually before patching.'
        )

    patched = text
    patched = replace_once(patched, CLI_ANCHOR, CLI_INSERT, 'CLI')
    patched = replace_once(
        patched,
        PROVENANCE_ANCHOR,
        PROVENANCE_INSERT,
        'Comp5 provenance',
    )
    patched = replace_once(
        patched,
        OPTIMIZER_ANCHOR,
        OPTIMIZER_INSERT,
        'optimizer construction',
    )
    patched = replace_once(
        patched,
        SCHEDULER_AUDIT_ANCHOR,
        SCHEDULER_AUDIT_INSERT,
        'scheduler audit',
    )

    compile(patched, str(runner), 'exec')

    shutil.copy2(runner, backup)
    runner.write_text(patched, encoding='utf-8')

    verify = runner.read_text(encoding='utf-8')
    compile(verify, str(runner), 'exec')

    print(f'Patched: {runner}')
    print(f'Backup : {backup}')
    print('Syntax check: PASS')
    print('Stage-6 optimizer path: rest base-LR + pretrained Comp5 differential-LR')


if __name__ == '__main__':
    main()

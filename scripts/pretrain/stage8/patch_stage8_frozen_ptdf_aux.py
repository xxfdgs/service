#!/usr/bin/env python3
"""Install Stage-8 frozen PT-DF auxiliary Fifth branch.

Patches
-------
1. The registered OneHotEmbedGPS implementation:
   - retains the historical trainable/random `comp5_encoder`;
   - optionally adds `frozen_comp5_aux_encoder`;
   - concatenates its pure pooled structural embedding into the fusion input;
   - keeps the auxiliary encoder permanently in eval mode;
   - excludes all frozen-branch parameters from gradients.

2. scripts/diagnostics/run_fusion_head_experiment.py:
   - adds --frozen-comp5-aux-checkpoint / --frozen-comp5-aux-label;
   - writes the enabling flag into the effective cfg before model creation;
   - strictly loads the Stage-4 checkpoint into the frozen auxiliary encoder;
   - audits requires_grad=False and exact key/shape compatibility;
   - writes frozen_comp5_aux_initialization.json;
   - adds frozen-branch provenance to selected checkpoints.

The patch is fail-closed and idempotent.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MODEL_MARKER = "frozen_comp5_aux_encoder"
RUNNER_MARKER = "--frozen-comp5-aux-checkpoint"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected anchor exactly once, found {count}. "
            "Refusing blind patch."
        )
    return text.replace(old, new, 1)


def discover_onehot(repo: Path) -> Path:
    candidates = []
    for path in repo.rglob("onehot_embed_gps.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if (
            "@register_network('OneHotEmbedGPS')" in text
            and "class Comp5GraphEncoder" in text
        ):
            candidates.append(path.resolve())

    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one registered onehot_embed_gps.py, found "
            f"{len(candidates)}: {candidates}"
        )
    return candidates[0]


# ============================================================================
# OneHotEmbedGPS patch
# ============================================================================

MODEL_ENCODER_OLD = """        # --- Pathway B: GraphGPS encoder for component 5 ---
        self.comp5_encoder = Comp5GraphEncoder(dim_in)
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
"""

MODEL_ENCODER_NEW = """        # --- Pathway B: task-specific GraphGPS encoder for component 5 ---
        #
        # Historical behavior is preserved when frozen_comp5_aux_enable=False:
        # this is the sole Fifth structural branch and is fully trainable.
        self.comp5_encoder = Comp5GraphEncoder(dim_in)
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

        # Stage-8 optional structural-prior branch.
        #
        # It is intentionally a SECOND encoder rather than an initialization of
        # comp5_encoder.  The downstream task branch therefore remains free to
        # learn Norm-specific structure while the Stage-4 PT-DF geometry is
        # preserved as an immutable auxiliary signal.
        self.frozen_comp5_aux_enable = bool(
            getattr(cfg.model, 'frozen_comp5_aux_enable', False)
        )
        self.frozen_comp5_aux_dim = (
            self.hidden_dim if self.frozen_comp5_aux_enable else 0
        )
        if self.frozen_comp5_aux_enable:
            self.frozen_comp5_aux_encoder = Comp5GraphEncoder(dim_in)
            for parameter in self.frozen_comp5_aux_encoder.parameters():
                parameter.requires_grad_(False)
            # BatchNorm running statistics must also remain frozen.
            self.frozen_comp5_aux_encoder.eval()
"""

MODEL_FUSION_DIM_OLD = """        if self.fifth_only_fusion:
            fusion_in_dim = self.hidden_dim + 1 + self.mordred_feature_dim
        else:
            fusion_in_dim = (
                self.num_components * self.hidden_dim + self.num_components
                + self.ratio_basis_dim
                + mordred_multiplier * self.mordred_feature_dim)
"""

MODEL_FUSION_DIM_NEW = """        if self.fifth_only_fusion:
            fusion_in_dim = (
                self.hidden_dim
                + self.frozen_comp5_aux_dim
                + 1
                + self.mordred_feature_dim
            )
        else:
            fusion_in_dim = (
                self.num_components * self.hidden_dim
                + self.frozen_comp5_aux_dim
                + self.num_components
                + self.ratio_basis_dim
                + mordred_multiplier * self.mordred_feature_dim
            )
"""

MODEL_TRAIN_ANCHOR = """    @staticmethod
    def _build_ratio_feature(ratio):
"""

MODEL_TRAIN_INSERT = """    def train(self, mode=True):
        # nn.Module.train(True) would otherwise put the frozen auxiliary
        # encoder and its BatchNorm layers back into training mode.  Re-lock
        # that branch after every train/eval transition.
        super().train(mode)
        if self.frozen_comp5_aux_enable:
            self.frozen_comp5_aux_encoder.eval()
        return self

    @staticmethod
    def _build_ratio_feature(ratio):
"""

MODEL_FORWARD_ENCODER_OLD = """        # --- Pathway B: GraphGPS for component 5 ---
        data5_encoded = self.comp5_encoder(data5)
        emb5 = self.pooling_fun(data5_encoded.x, data5_encoded.batch)  # [B, hidden_dim]
"""

MODEL_FORWARD_ENCODER_NEW = """        # --- Pathway B: GraphGPS for component 5 ---
        #
        # Comp5GraphEncoder mutates the PyG batch while encoding.  The frozen
        # structural-prior branch must therefore receive a clone of the raw
        # Fifth batch, not the task encoder's already-transformed object.
        emb5_frozen = None
        if self.frozen_comp5_aux_enable:
            frozen_input = data5.clone()
            self.frozen_comp5_aux_encoder.eval()
            with torch.no_grad():
                frozen_encoded = self.frozen_comp5_aux_encoder(frozen_input)
                emb5_frozen = self.pooling_fun(
                    frozen_encoded.x, frozen_encoded.batch
                )
            if emb5_frozen.size(-1) != self.hidden_dim:
                raise RuntimeError(
                    'Frozen Comp5 auxiliary embedding dimension mismatch: '
                    f'{emb5_frozen.size(-1)} != {self.hidden_dim}'
                )

        data5_encoded = self.comp5_encoder(data5)
        emb5 = self.pooling_fun(data5_encoded.x, data5_encoded.batch)  # [B, hidden_dim]
"""

MODEL_FUSION_PARTS_OLD = """        if self.fifth_only_fusion:
            fusion_parts = [emb5, ratio5]
            if cfg.use_mordred_features:
                fusion_parts.append(
                    data5.mordred_feat.view(
                        data5.num_graphs, -1).float())
        else:
            fusion_parts = [all_embs, all_ratios]
            if self.ratio_polynomial_features:
"""

MODEL_FUSION_PARTS_NEW = """        if self.fifth_only_fusion:
            fusion_parts = [emb5]
            if self.frozen_comp5_aux_enable:
                fusion_parts.append(emb5_frozen)
            fusion_parts.append(ratio5)
            if cfg.use_mordred_features:
                fusion_parts.append(
                    data5.mordred_feat.view(
                        data5.num_graphs, -1).float())
        else:
            # Keep all_embs as fusion_parts[0] because gated/attention fusion
            # rewrites exactly that component-token block below.
            fusion_parts = [all_embs]
            if self.frozen_comp5_aux_enable:
                fusion_parts.append(emb5_frozen)
            fusion_parts.append(all_ratios)
            if self.ratio_polynomial_features:
"""


def patch_model(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if MODEL_MARKER in text:
        compile(text, str(path), "exec")
        print(f"[model] Stage-8 patch already present: {path}")
        return

    patched = text
    patched = replace_once(
        patched, MODEL_ENCODER_OLD, MODEL_ENCODER_NEW,
        "model encoder construction"
    )
    patched = replace_once(
        patched, MODEL_FUSION_DIM_OLD, MODEL_FUSION_DIM_NEW,
        "model fusion dimension"
    )
    patched = replace_once(
        patched, MODEL_TRAIN_ANCHOR, MODEL_TRAIN_INSERT,
        "model train override"
    )
    patched = replace_once(
        patched, MODEL_FORWARD_ENCODER_OLD, MODEL_FORWARD_ENCODER_NEW,
        "model Fifth forward"
    )
    patched = replace_once(
        patched, MODEL_FUSION_PARTS_OLD, MODEL_FUSION_PARTS_NEW,
        "model fusion parts"
    )

    compile(patched, str(path), "exec")

    backup = path.with_name(path.name + ".pre_stage8_frozen_aux_backup")
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")

    print(f"[model] patched: {path}")
    print(f"[model] backup : {backup}")


# ============================================================================
# Runner patch
# ============================================================================

RUNNER_CLI_ANCHOR = """    parser.add_argument('--resume', action='store_true')
"""

RUNNER_CLI_INSERT = """    parser.add_argument(
        '--frozen-comp5-aux-checkpoint',
        type=Path,
        default=None,
        help=(
            'Optional Stage-4 Comp5GraphEncoder checkpoint loaded into a '
            'second, permanently frozen Fifth structural-prior branch. '
            'The historical comp5_encoder remains a separate trainable branch.'
        ),
    )
    parser.add_argument(
        '--frozen-comp5-aux-label',
        type=str,
        default='none',
        help='Audit label for the frozen Fifth structural-prior branch.',
    )
    parser.add_argument('--resume', action='store_true')
"""

RUNNER_CFG_ANCHOR = """    cfg.model.target_specific_heads = args.head_type == 'target_specific'
    cfg.model.validate_redesign_inputs = args.candidate != 'A0'
"""

RUNNER_CFG_INSERT = """    cfg.model.target_specific_heads = args.head_type == 'target_specific'

    # Stage-8 frozen structural-prior branch is a runtime architecture flag.
    # Register it before cache/effective-config persistence so external
    # inference can reconstruct exactly the same model topology.
    cfg.model.set_new_allowed(True)
    try:
        cfg.model.frozen_comp5_aux_enable = bool(
            args.frozen_comp5_aux_checkpoint is not None
        )
    finally:
        cfg.model.set_new_allowed(False)

    cfg.model.validate_redesign_inputs = args.candidate != 'A0'
"""

RUNNER_MODEL_ANCHOR = """    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model = create_model_gps().to(device)
"""

RUNNER_MODEL_INSERT = """    device = torch.device(cfg.accelerator, cfg.gpu_serial)
    model = create_model_gps().to(device)

    # ------------------------------------------------------------------
    # Stage-8 frozen PT-DF structural-prior branch.
    # ------------------------------------------------------------------
    frozen_comp5_aux_metadata = {
        'enabled': bool(args.frozen_comp5_aux_checkpoint is not None),
        'label': str(args.frozen_comp5_aux_label),
        'checkpoint': None,
        'checkpoint_sha256': None,
        'strict': False,
        'loaded_tensors': 0,
        'trainable_parameter_count': 0,
        'frozen_parameter_count': 0,
    }

    if args.frozen_comp5_aux_checkpoint is not None:
        frozen_path = args.frozen_comp5_aux_checkpoint.resolve()
        if not frozen_path.is_file():
            raise FileNotFoundError(
                f'Frozen Comp5 auxiliary checkpoint is missing: {frozen_path}'
            )
        if str(cfg.model.type) != 'OneHotEmbedGPS':
            raise ValueError(
                '--frozen-comp5-aux-checkpoint requires OneHotEmbedGPS; '
                f'effective model.type={cfg.model.type!r}'
            )
        if not hasattr(model, 'model'):
            raise AttributeError(
                'GraphGym wrapper has no `.model` core target.'
            )
        core = model.model
        if not getattr(core, 'frozen_comp5_aux_enable', False):
            raise RuntimeError(
                'Frozen Comp5 checkpoint requested but model topology did not '
                'enable frozen_comp5_aux_encoder.'
            )
        if not hasattr(core, 'frozen_comp5_aux_encoder'):
            raise AttributeError(
                'OneHotEmbedGPS core has no frozen_comp5_aux_encoder.'
            )

        payload = torch.load(
            frozen_path,
            map_location='cpu',
            weights_only=False,
        )
        if isinstance(payload, dict) and 'encoder_state_dict' in payload:
            source_state = payload['encoder_state_dict']
        else:
            source_state = payload

        if not isinstance(source_state, dict):
            raise TypeError(
                'Frozen Comp5 auxiliary checkpoint does not contain a state_dict.'
            )

        # Accept the Stage-4 prefixed transfer artifact as well as the raw one.
        if source_state and all(
            str(key).startswith('comp5_encoder.')
            for key in source_state
        ):
            source_state = {
                str(key)[len('comp5_encoder.'):]: value
                for key, value in source_state.items()
            }

        target_state = core.frozen_comp5_aux_encoder.state_dict()
        source_keys = set(source_state)
        target_keys = set(target_state)
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        shape_mismatches = {
            key: {
                'source': tuple(source_state[key].shape),
                'target': tuple(target_state[key].shape),
            }
            for key in sorted(source_keys & target_keys)
            if tuple(source_state[key].shape)
            != tuple(target_state[key].shape)
        }
        if missing or unexpected or shape_mismatches:
            raise RuntimeError(
                'Frozen Stage-4 encoder is not interface-compatible.\\n'
                f'missing={missing}\\n'
                f'unexpected={unexpected}\\n'
                f'shape_mismatches={shape_mismatches}'
            )

        core.frozen_comp5_aux_encoder.load_state_dict(
            source_state,
            strict=True,
        )
        for parameter in core.frozen_comp5_aux_encoder.parameters():
            parameter.requires_grad_(False)
        core.frozen_comp5_aux_encoder.eval()

        trainable_frozen = int(sum(
            parameter.numel()
            for parameter in core.frozen_comp5_aux_encoder.parameters()
            if parameter.requires_grad
        ))
        frozen_count = int(sum(
            parameter.numel()
            for parameter in core.frozen_comp5_aux_encoder.parameters()
        ))
        if trainable_frozen != 0:
            raise RuntimeError(
                'Frozen Comp5 auxiliary encoder still has trainable parameters.'
            )

        # The task branch must remain a separate trainable encoder.
        task_trainable = int(sum(
            parameter.numel()
            for parameter in core.comp5_encoder.parameters()
            if parameter.requires_grad
        ))
        if task_trainable <= 0:
            raise RuntimeError(
                'Stage-8 task-specific comp5_encoder is unexpectedly frozen.'
            )

        frozen_comp5_aux_metadata.update({
            'checkpoint': str(frozen_path),
            'checkpoint_sha256': file_sha256(frozen_path),
            'strict': True,
            'loaded_tensors': len(source_state),
            'trainable_parameter_count': trainable_frozen,
            'frozen_parameter_count': frozen_count,
            'task_comp5_trainable_parameter_count': task_trainable,
        })

        print(
            '[Stage8] Frozen Comp5 auxiliary transfer PASS: '
            f'{frozen_path}'
        )
        print(
            '[Stage8] Branch audit PASS: '
            f'frozen_params={frozen_count}, '
            f'frozen_trainable=0, task_comp5_trainable={task_trainable}'
        )

    (run_dir / 'frozen_comp5_aux_initialization.json').write_text(
        json.dumps(frozen_comp5_aux_metadata, indent=2) + '\\n',
        encoding='utf-8',
    )
"""

RUNNER_META_ANCHOR = """    checkpoint_metadata['comp5_initialization'] = comp5_init_metadata
    start_epoch, best_loss, best_epoch, best_state = 0, math.inf, None, None
"""

RUNNER_META_INSERT = """    checkpoint_metadata['comp5_initialization'] = comp5_init_metadata
    checkpoint_metadata[
        'frozen_comp5_aux_initialization'
    ] = frozen_comp5_aux_metadata
    start_epoch, best_loss, best_epoch, best_state = 0, math.inf, None, None
"""


def patch_runner(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if RUNNER_MARKER in text:
        compile(text, str(path), "exec")
        print(f"[runner] Stage-8 patch already present: {path}")
        return

    required = [
        "--comp5-pretrained-checkpoint",
        "comp5_initialization.json",
    ]
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(
            "Current runner lacks Stage-5 transfer prerequisites: "
            + ", ".join(missing)
        )

    patched = text
    patched = replace_once(
        patched, RUNNER_CLI_ANCHOR, RUNNER_CLI_INSERT,
        "runner CLI"
    )
    patched = replace_once(
        patched, RUNNER_CFG_ANCHOR, RUNNER_CFG_INSERT,
        "runner cfg"
    )
    patched = replace_once(
        patched, RUNNER_MODEL_ANCHOR, RUNNER_MODEL_INSERT,
        "runner model construction"
    )
    patched = replace_once(
        patched, RUNNER_META_ANCHOR, RUNNER_META_INSERT,
        "runner checkpoint metadata"
    )

    compile(patched, str(path), "exec")

    backup = path.with_name(path.name + ".pre_stage8_frozen_aux_backup")
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")

    print(f"[runner] patched: {path}")
    print(f"[runner] backup : {backup}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/diagnostics/run_fusion_head_experiment.py"),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    runner = (repo / args.runner).resolve()
    if not runner.is_file():
        raise FileNotFoundError(runner)

    model = discover_onehot(repo)

    patch_model(model)
    patch_runner(runner)

    print()
    print("Stage-8 frozen PT-DF auxiliary-branch patch: PASS")
    print(f"OneHotEmbedGPS: {model}")
    print(f"Runner        : {runner}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
'''Stage-8 v2 patcher: frozen PT-DF auxiliary Fifth branch.

This version avoids replacing the entire historical fusion_in_dim formula.
Instead it preserves whatever fusion features the current local model already
has and only appends:

    fusion_in_dim += self.frozen_comp5_aux_dim

Likewise, the frozen embedding is inserted into the already-built fusion_parts
immediately before the first torch.cat().  This is robust to later additions
to ratio/Mordred/auxiliary fusion features.

The patch remains fail-closed and transactional: files are written only after
ALL in-memory edits compile successfully.
'''

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MODEL_MARKER = "frozen_comp5_aux_encoder"
RUNNER_MARKER = "--frozen-comp5-aux-checkpoint"


def require_once(text: str, anchor: str, label: str) -> int:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected anchor exactly once, found {count}. "
            "Refusing blind patch."
        )
    return text.index(anchor)


def insert_before_once(text: str, anchor: str, insertion: str, label: str) -> str:
    pos = require_once(text, anchor, label)
    return text[:pos] + insertion + text[pos:]


def insert_after_once(text: str, anchor: str, insertion: str, label: str) -> str:
    pos = require_once(text, anchor, label)
    end = pos + len(anchor)
    return text[:end] + insertion + text[end:]


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


# =============================================================================
# Model patch
# =============================================================================

MODEL_POOLING_ANCHOR = (
    "        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]\\n"
)

MODEL_ENCODER_INSERT = r'''
        # Stage-8 optional frozen structural-prior branch.
        #
        # Historical `comp5_encoder` remains the independent task-specific
        # encoder. This second encoder is loaded from Stage-4 PT-DF and is
        # permanently frozen.
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
            # BatchNorm running statistics must remain fixed as well.
            self.frozen_comp5_aux_encoder.eval()
'''

MODEL_DROPOUT_ANCHOR = (
    "        dropout = float(getattr(cfg.gt, 'dropout', 0.0))\\n"
)

MODEL_FUSION_DIM_INSERT = r'''        # Stage-8: preserve the complete historical fusion formula and
        # append exactly one hidden_dim-sized frozen structural embedding.
        fusion_in_dim += self.frozen_comp5_aux_dim
'''

MODEL_TRAIN_ANCHOR = (
    "    @staticmethod\\n"
    "    def _build_ratio_feature(ratio):\\n"
)

MODEL_TRAIN_INSERT = r'''    def train(self, mode=True):
        # nn.Module.train(True) recursively switches all children to train
        # mode. Re-lock the frozen structural-prior branch so BatchNorm
        # statistics and dropout behavior remain identical to Stage-4 eval.
        super().train(mode)
        if self.frozen_comp5_aux_enable:
            self.frozen_comp5_aux_encoder.eval()
        return self

'''

MODEL_TASK_FORWARD_ANCHOR = (
    "        data5_encoded = self.comp5_encoder(data5)\\n"
)

MODEL_FROZEN_FORWARD_INSERT = r'''        # Stage-8 frozen PT-DF structural prior.
        #
        # Comp5GraphEncoder mutates the PyG Batch object during encoding, so
        # the frozen branch must receive an independent clone of the raw Fifth
        # graph before the trainable task encoder consumes `data5`.
        emb5_frozen = None
        if self.frozen_comp5_aux_enable:
            frozen_input = data5.clone()
            self.frozen_comp5_aux_encoder.eval()
            with torch.no_grad():
                frozen_encoded = self.frozen_comp5_aux_encoder(frozen_input)
                emb5_frozen = self.pooling_fun(
                    frozen_encoded.x,
                    frozen_encoded.batch,
                )
            if emb5_frozen.ndim != 2:
                raise RuntimeError(
                    'Frozen Comp5 auxiliary embedding must be rank-2; '
                    f'got shape={tuple(emb5_frozen.shape)}'
                )
            if emb5_frozen.size(-1) != self.hidden_dim:
                raise RuntimeError(
                    'Frozen Comp5 auxiliary embedding dimension mismatch: '
                    f'{emb5_frozen.size(-1)} != {self.hidden_dim}'
                )

'''

MODEL_FIRST_COMBINED_ANCHOR = (
    "        combined = torch.cat(fusion_parts, dim=1)\\n"
)

MODEL_FUSION_PARTS_INSERT = r'''        # Stage-8: append the frozen pure-structure signal without changing
        # the historical meaning of fusion_parts[0].
        #
        # Keeping fusion_parts[0] untouched is important because gated_concat
        # and attention_concat later rewrite that element as the component
        # token block. Insert the frozen vector immediately after it.
        if self.frozen_comp5_aux_enable:
            if emb5_frozen is None:
                raise RuntimeError(
                    'Frozen Comp5 auxiliary branch is enabled but no embedding '
                    'was produced.'
                )
            fusion_parts.insert(1, emb5_frozen)

'''


def patch_model_text(text: str) -> str:
    if MODEL_MARKER in text:
        compile(text, "<already-patched-onehot>", "exec")
        return text

    required = [
        "self.comp5_encoder = Comp5GraphEncoder(dim_in)",
        MODEL_POOLING_ANCHOR.strip(),
        "fusion_in_dim",
        MODEL_DROPOUT_ANCHOR.strip(),
        "def forward(self, data1, data2, data3, data4, data5):",
        MODEL_TASK_FORWARD_ANCHOR.strip(),
        "fusion_parts",
        MODEL_FIRST_COMBINED_ANCHOR.strip(),
    ]
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(
            "OneHotEmbedGPS no longer satisfies Stage-8 prerequisites; "
            "missing markers: " + ", ".join(missing)
        )

    patched = text

    patched = insert_after_once(
        patched,
        MODEL_POOLING_ANCHOR,
        MODEL_ENCODER_INSERT,
        "model pooling/encoder insertion",
    )

    patched = insert_before_once(
        patched,
        MODEL_DROPOUT_ANCHOR,
        MODEL_FUSION_DIM_INSERT,
        "model fusion-dimension insertion",
    )

    patched = insert_before_once(
        patched,
        MODEL_TRAIN_ANCHOR,
        MODEL_TRAIN_INSERT,
        "model train override",
    )

    patched = insert_before_once(
        patched,
        MODEL_TASK_FORWARD_ANCHOR,
        MODEL_FROZEN_FORWARD_INSERT,
        "model frozen forward",
    )

    # The model can have more than one `combined = torch.cat(...)` because
    # gated/attention fusion reconstruct `combined`. Insert before the FIRST
    # one inside forward.
    forward_pos = patched.index(
        "    def forward(self, data1, data2, data3, data4, data5):"
    )
    combined_pos = patched.find(
        MODEL_FIRST_COMBINED_ANCHOR,
        forward_pos,
    )
    if combined_pos < 0:
        raise RuntimeError(
            "Could not find first fusion_parts concatenation inside forward."
        )
    patched = (
        patched[:combined_pos]
        + MODEL_FUSION_PARTS_INSERT
        + patched[combined_pos:]
    )

    expected_once = [
        "self.frozen_comp5_aux_enable = bool(",
        "self.frozen_comp5_aux_encoder = Comp5GraphEncoder(dim_in)",
        "fusion_in_dim += self.frozen_comp5_aux_dim",
        "def train(self, mode=True):",
        "frozen_input = data5.clone()",
        "fusion_parts.insert(1, emb5_frozen)",
    ]
    for marker in expected_once:
        count = patched.count(marker)
        if count != 1:
            raise RuntimeError(
                f"Post-patch invariant failed for {marker!r}: count={count}"
            )

    compile(patched, "<patched-onehot>", "exec")
    return patched


# =============================================================================
# Runner patch
# =============================================================================

RUNNER_CLI_ANCHOR = (
    "    parser.add_argument('--resume', action='store_true')\\n"
)

RUNNER_CLI_INSERT = r'''    parser.add_argument(
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
'''

RUNNER_CFG_ANCHOR = (
    "    cfg.model.target_specific_heads = args.head_type == 'target_specific'\\n"
)

RUNNER_CFG_INSERT = r'''
    # Stage-8 runtime topology flag. Set before model construction and before
    # effective_config persistence so external inference can reconstruct the
    # same architecture.
    cfg.model.set_new_allowed(True)
    try:
        cfg.model.frozen_comp5_aux_enable = bool(
            args.frozen_comp5_aux_checkpoint is not None
        )
    finally:
        cfg.model.set_new_allowed(False)
'''

RUNNER_MODEL_ANCHOR = (
    "    model = create_model_gps().to(device)\\n"
)

RUNNER_MODEL_INSERT = r'''
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
                'Frozen checkpoint requested but model topology did not enable '
                'frozen_comp5_aux_encoder.'
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
                'Frozen Comp5 checkpoint does not contain a state_dict.'
            )

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
        missing_keys = sorted(target_keys - source_keys)
        unexpected_keys = sorted(source_keys - target_keys)
        shape_mismatches = {
            key: {
                'source': tuple(source_state[key].shape),
                'target': tuple(target_state[key].shape),
            }
            for key in sorted(source_keys & target_keys)
            if tuple(source_state[key].shape)
            != tuple(target_state[key].shape)
        }

        if missing_keys or unexpected_keys or shape_mismatches:
            raise RuntimeError(
                'Frozen Stage-4 encoder is not interface-compatible.\\n'
                f'missing={missing_keys}\\n'
                f'unexpected={unexpected_keys}\\n'
                f'shape_mismatches={shape_mismatches}'
            )

        core.frozen_comp5_aux_encoder.load_state_dict(
            source_state,
            strict=True,
        )
        for parameter in core.frozen_comp5_aux_encoder.parameters():
            parameter.requires_grad_(False)
        core.frozen_comp5_aux_encoder.eval()

        frozen_count = int(sum(
            parameter.numel()
            for parameter in core.frozen_comp5_aux_encoder.parameters()
        ))
        frozen_trainable = int(sum(
            parameter.numel()
            for parameter in core.frozen_comp5_aux_encoder.parameters()
            if parameter.requires_grad
        ))
        task_trainable = int(sum(
            parameter.numel()
            for parameter in core.comp5_encoder.parameters()
            if parameter.requires_grad
        ))

        if frozen_trainable != 0:
            raise RuntimeError(
                'Frozen Comp5 auxiliary encoder still has trainable parameters.'
            )
        if task_trainable <= 0:
            raise RuntimeError(
                'Task-specific comp5_encoder is unexpectedly frozen.'
            )

        frozen_comp5_aux_metadata.update({
            'checkpoint': str(frozen_path),
            'checkpoint_sha256': file_sha256(frozen_path),
            'strict': True,
            'loaded_tensors': len(source_state),
            'trainable_parameter_count': frozen_trainable,
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
            f'frozen_trainable={frozen_trainable}, '
            f'task_comp5_trainable={task_trainable}'
        )

    (run_dir / 'frozen_comp5_aux_initialization.json').write_text(
        json.dumps(frozen_comp5_aux_metadata, indent=2) + '\\n',
        encoding='utf-8',
    )
'''

RUNNER_META_ANCHOR = (
    "    checkpoint_metadata['comp5_initialization'] = comp5_init_metadata\\n"
)

RUNNER_META_INSERT = r'''    checkpoint_metadata[
        'frozen_comp5_aux_initialization'
    ] = frozen_comp5_aux_metadata
'''


def patch_runner_text(text: str) -> str:
    if RUNNER_MARKER in text:
        compile(text, "<already-patched-runner>", "exec")
        return text

    required = [
        "--comp5-pretrained-checkpoint",
        "comp5_initialization.json",
        "load_stage4_comp5_encoder",
        RUNNER_CLI_ANCHOR.strip(),
        RUNNER_CFG_ANCHOR.strip(),
        RUNNER_MODEL_ANCHOR.strip(),
        RUNNER_META_ANCHOR.strip(),
    ]
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(
            "Current runner no longer satisfies Stage-8 prerequisites; "
            "missing markers: " + ", ".join(missing)
        )

    patched = text

    patched = insert_before_once(
        patched,
        RUNNER_CLI_ANCHOR,
        RUNNER_CLI_INSERT,
        "runner CLI insertion",
    )

    patched = insert_after_once(
        patched,
        RUNNER_CFG_ANCHOR,
        RUNNER_CFG_INSERT,
        "runner cfg insertion",
    )

    patched = insert_after_once(
        patched,
        RUNNER_MODEL_ANCHOR,
        RUNNER_MODEL_INSERT,
        "runner model insertion",
    )

    patched = insert_after_once(
        patched,
        RUNNER_META_ANCHOR,
        RUNNER_META_INSERT,
        "runner checkpoint metadata insertion",
    )

    expected = [
        "--frozen-comp5-aux-checkpoint",
        "cfg.model.frozen_comp5_aux_enable = bool(",
        "Frozen Comp5 auxiliary transfer PASS",
        "'frozen_comp5_aux_initialization'",
    ]
    for marker in expected:
        if marker not in patched:
            raise RuntimeError(
                f"Runner post-patch invariant missing marker: {marker!r}"
            )

    compile(patched, "<patched-runner>", "exec")
    return patched


# =============================================================================
# Transactional update
# =============================================================================

def transactional_write(path: Path, patched: str, backup_suffix: str) -> None:
    original = path.read_text(encoding="utf-8")
    if patched == original:
        print(f"[skip] already patched: {path}")
        return

    backup = path.with_name(path.name + backup_suffix)
    if backup.exists():
        raise FileExistsError(
            f"Backup already exists: {backup}. Refusing to overwrite it."
        )

    compile(patched, str(path), "exec")
    shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")

    verify = path.read_text(encoding="utf-8")
    compile(verify, str(path), "exec")

    print(f"[patched] {path}")
    print(f"[backup ] {backup}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/diagnostics/run_fusion_head_experiment.py"),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    model_path = discover_onehot(repo)
    runner_path = (repo / args.runner).resolve()

    if not runner_path.is_file():
        raise FileNotFoundError(runner_path)

    model_original = model_path.read_text(encoding="utf-8")
    runner_original = runner_path.read_text(encoding="utf-8")

    # Build and syntax-check BOTH modified files before writing either one.
    model_patched = patch_model_text(model_original)
    runner_patched = patch_runner_text(runner_original)

    print("[audit] in-memory model patch syntax: PASS")
    print("[audit] in-memory runner patch syntax: PASS")

    transactional_write(
        model_path,
        model_patched,
        ".pre_stage8_v2_backup",
    )
    transactional_write(
        runner_path,
        runner_patched,
        ".pre_stage8_v2_backup",
    )

    model_final = model_path.read_text(encoding="utf-8")
    runner_final = runner_path.read_text(encoding="utf-8")

    model_required = [
        "frozen_comp5_aux_encoder",
        "fusion_in_dim += self.frozen_comp5_aux_dim",
        "frozen_input = data5.clone()",
        "fusion_parts.insert(1, emb5_frozen)",
    ]
    runner_required = [
        "--frozen-comp5-aux-checkpoint",
        "frozen_comp5_aux_initialization.json",
        "Frozen Comp5 auxiliary transfer PASS",
    ]

    missing_model = [x for x in model_required if x not in model_final]
    missing_runner = [x for x in runner_required if x not in runner_final]

    if missing_model or missing_runner:
        raise RuntimeError(
            "Final Stage-8 marker audit failed: "
            f"model_missing={missing_model}, "
            f"runner_missing={missing_runner}"
        )

    print()
    print("=" * 80)
    print("STAGE-8 V2 PATCH PASS")
    print("=" * 80)
    print(f"OneHotEmbedGPS: {model_path}")
    print(f"Runner        : {runner_path}")
    print("Fusion policy : preserve existing fusion formula + frozen hidden vector")
    print("Write policy  : transactional / fail-closed")


if __name__ == "__main__":
    main()

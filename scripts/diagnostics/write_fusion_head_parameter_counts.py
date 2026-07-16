#!/usr/bin/env python3
"""Write parameter counts for every planned fusion/head architecture."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import graphgps  # noqa: E402,F401
from graphgps.config.config_gps import set_cfg_gps  # noqa: E402
from graphgps.create_model_gps import create_model_gps  # noqa: E402
from loader_5 import create_loader_5  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402


CONFIG = ROOT / 'results/deduplicated_rebaseline/graphgps_cv/configs/formula_identity_group_cv_fold_0_seed_0.yaml'
OUTPUT = ROOT / 'results/fusion_head_redesign_exp/implementation/parameter_counts.csv'


def count(group, candidate, fusion_type, head_type):
    cfg.model.fusion_type = fusion_type
    cfg.model.head_type = head_type
    cfg.model.architecture_name = ('legacy_baseline' if candidate == 'A0'
                                   else f'{group}_{candidate}_{fusion_type}_{head_type}')
    cfg.model.validate_redesign_inputs = candidate != 'A0'
    cfg.accelerator = 'cpu'
    cfg.gpu_serial = 0
    model = create_model_gps()
    return {'group': group, 'candidate': candidate, 'architecture': cfg.model.architecture_name,
            'fusion_type': fusion_type, 'head_type': head_type,
            'core_model_parameter_count': sum(p.numel() for p in model.model.parameters()),
            'state_dict_numel': sum(v.numel() for v in model.state_dict().values())}


def main():
    set_cfg_gps(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(CONFIG), opts=[]))
    cfg.dataset.dir = str(ROOT / 'results/fusion_head_redesign_exp/stage1/group_a/A0/fold_0/cache')
    cfg.dataset.cache_refresh = False
    cfg.accelerator = 'cpu'
    cfg.gpu_serial = 0
    # Loader setup configures positional-encoding dimensions from actual data;
    # otherwise an empty PE placeholder would under-count checkpoint tensors.
    create_loader_5()
    rows = [count('A', 'A0', 'softmax_sum', 'baseline')]
    rows += [count('A', candidate, 'softmax_sum', head) for candidate, head in (
        ('A1', 'linear'), ('A2', 'two_layer'), ('A3', 'residual_head'),
        ('A4', 'target_specific'))]
    rows += [count('B', candidate, fusion, 'linear') for candidate, fusion in (
        ('B0', 'softmax_sum'), ('B1', 'concat'), ('B2', 'concat_mlp'),
        ('B3', 'residual'), ('B4', 'gated_concat'))]
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()

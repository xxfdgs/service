#!/usr/bin/env python3
"""CPU checks for fusion/head shape, deterministic and numerical invariants."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from graphgps.network.double_gps_cat_v31_muliti_4_v0 import (  # noqa: E402
    GPSModel, RedesignFusion, RedesignHead)


class FusionHeadInterfaceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.graph = torch.randn(5, 12)
        self.descriptor = torch.randn(5, 8)
        self.formula = torch.tensor([
            [0.2, 0.1, 0., 0.7], [0.1, 0.2, 0.3, 0.4],
            [0., 0., 0., 1.], [0.25, 0.25, 0.25, 0.25],
            [1., 0., 0., 0.],
        ])

    def test_fusion_shapes_are_finite_and_deterministic(self):
        for fusion_type in ('softmax_sum', 'concat', 'concat_mlp', 'residual', 'gated_concat'):
            module = RedesignFusion(12, 8, 4, 6, fusion_type).eval()
            first, diagnostics = module(self.graph, self.descriptor, self.formula)
            second, _ = module(self.graph, self.descriptor, self.formula)
            self.assertEqual(tuple(first.shape), (5, module.output_dim))
            self.assertTrue(torch.isfinite(first).all())
            self.assertTrue(torch.equal(first, second))
            self.assertGreater(float(diagnostics['graph_branch'].std()), 0.0)

    def test_all_heads_shape_and_no_nan(self):
        features = torch.randn(5, 18)
        for head_type in ('linear', 'two_layer', 'residual_head', 'target_specific'):
            head = RedesignHead(18, 4, head_type, 8).eval()
            output, diagnostics = head(features)
            self.assertEqual(tuple(output.shape), (5, 4))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(tuple(diagnostics['head_input'].shape), (5, 18))

    def test_missing_component_and_non_100_ratio_are_safe(self):
        # _ratio_features is independent from model parameters; zero entries
        # emulate absent components and ratios summing to 1.2 emulate a
        # malformed non-100% formulation. Values must remain finite/clamped.
        ratio = torch.tensor([0., 0.2, 1.2])
        features = GPSModel._ratio_features(None, ratio)
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(float(features[0, 3]), 0.0)
        self.assertEqual(float(features[2, 0]), 1.0)


if __name__ == '__main__':
    unittest.main()

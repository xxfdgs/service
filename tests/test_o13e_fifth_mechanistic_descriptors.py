"""Minimal structural tests for the deterministic O13-E descriptor rules."""

import unittest

import numpy as np

from graphgps.lrx_add.fifth_mechanistic_descriptors import (
    MECHANISTIC_DESCRIPTOR_NAMES,
    descriptor_vector,
)


def values(smiles: str) -> dict[str, float]:
    return dict(zip(MECHANISTIC_DESCRIPTOR_NAMES, descriptor_vector(smiles)))


class TestO13EMechanisticDescriptors(unittest.TestCase):
    def test_single_tail(self):
        row = values("CCCCCCCCCCCCN")
        self.assertEqual(row["tail_count"], 1)
        self.assertGreaterEqual(row["max_tail_length"], 12)
        self.assertEqual(row["ionizable_N_count"], 1)

    def test_double_tail(self):
        row = values("CCCCCCCC(=O)OCCOC(=O)CCCCCCC")
        self.assertEqual(row["tail_count"], 2)
        self.assertEqual(row["ester_count"], 2)
        self.assertGreaterEqual(row["total_tail_carbon_count"], 14)

    def test_branched_tail(self):
        row = values("CCCCCC(C)CCCC")
        self.assertEqual(row["tail_count"], 1)
        self.assertGreater(row["branch_density"], 0)

    def test_unsaturated_tail(self):
        row = values("CCCCCC=CCCCCC")
        self.assertEqual(row["tail_count"], 1)
        self.assertEqual(row["double_bond_count"], 1)

    def test_ester_linker(self):
        row = values("CCCCCCCC(=O)OCCN")
        self.assertEqual(row["ester_count"], 1)
        self.assertEqual(row["ionizable_N_count"], 1)
        self.assertGreater(row["head_to_linker_distance"], 0)

    def test_absent_fr_is_zero(self):
        self.assertTrue(np.array_equal(
            descriptor_vector("[Fr]"),
            np.zeros(len(MECHANISTIC_DESCRIPTOR_NAMES), dtype=np.float32),
        ))


if __name__ == "__main__":
    unittest.main()

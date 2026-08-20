"""CPU-only structural contract tests for O13-F semantic Fifth features."""

import unittest

from graphgps.lrx_add.fifth_semantic_features import semantic_features


UC_LEU12_ESTER = "CCCCCCCCCCCCNC(=O)N[C@@H](C(=O)OCC)CC(C)C"
UC_ASP18_ACID = "CCCCCCCCCCCCCCCCCCNC(=O)N[C@H](C(O)=O)CC(=O)O"
DOPE_SS_PEPTIDE = (
    "CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@]([H])(OC(=O)CCCCCCC/C=C\\CCCCCCCC)"
    "COP(=O)(OCCNC(=O)CCSSCC(C(=O)O)NC(=O)C(CO)N)O"
)


class O13FSemanticFeatureTest(unittest.TestCase):
    def test_uc_type_tail_and_ester(self):
        result = semantic_features(UC_LEU12_ESTER)
        self.assertEqual(result.family_type, "UC_series")
        self.assertEqual(result.uc_amino_acid_type, "L")
        self.assertEqual(result.uc_tail_carbon_count, 12)
        self.assertEqual((result.uc_terminal_carboxyl, result.uc_terminal_ester), (0, 1))

    def test_uc_distinguishes_residue_tail_and_acid(self):
        result = semantic_features(UC_ASP18_ACID)
        self.assertEqual(result.family_type, "UC_series")
        self.assertEqual(result.uc_amino_acid_type, "D")
        self.assertEqual(result.uc_tail_carbon_count, 18)
        self.assertEqual(result.uc_terminal_carboxyl, 1)

    def test_dope_ss_peptide_requires_structure(self):
        result = semantic_features(DOPE_SS_PEPTIDE)
        self.assertEqual(result.family_type, "DOPE_SS_peptide_series")
        self.assertEqual(result.has_dope_tail, 1)
        self.assertEqual(result.disulfide_bridge_count, 1)
        self.assertGreaterEqual(result.peptide_length, 2)
        self.assertGreaterEqual(result.aa_counts["S"], 1)

    def test_real_disulfide_count_and_absence(self):
        self.assertEqual(semantic_features("CSSC").disulfide_bridge_count, 1)
        absent = semantic_features("[Fr]")
        self.assertEqual(absent.family_type, "other")
        self.assertEqual(absent.numeric_vector().sum(), 0)


if __name__ == "__main__":
    unittest.main()

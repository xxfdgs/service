"""
@Name:  graph_feature.py
@Auth:  rongxing
@Date:  2023/2/4-上午10:04
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME}
define the feature of node and edge
"""

from rdkit import Chem
import numpy as np

def smiles2graph(mol, coarse_grain=False, min_chain_length=6):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    # mol = Chem.MolFromSmiles('CCCCCCC1(CCCCCC)c2cc(*)ccc2-c2c1cc(cc2)c1ccc2-c3c([C]4(=CC=[C](C=C4)(C4=NC(CO4)c4ccccc4)C4=NC(CO4)c4ccccc4)c2c1)cc(cc3)*')
    if coarse_grain:
        return coarse_grained_smiles2graph(mol, min_chain_length)

    #### add H
    mol = Chem.AddHs(mol)

    # print('Chem.MolToSmiles(mol)=', Chem.MolToSmiles(mol))
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph


def find_linear_carbon_chains(mol, min_chain_length):
    """Find non-ring, unbranched sp3 carbon components eligible for merging."""
    candidates = {
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6
        and not atom.GetIsAromatic()
        and not atom.IsInRing()
        and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3
    }
    adjacency = {index: set() for index in candidates}
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (begin in candidates and end in candidates
                and bond.GetBondType() == Chem.rdchem.BondType.SINGLE):
            adjacency[begin].add(end)
            adjacency[end].add(begin)

    chains = []
    visited = set()
    for start in candidates:
        if start in visited:
            continue
        stack, component = [start], []
        visited.add(start)
        while stack:
            atom_index = stack.pop()
            component.append(atom_index)
            for neighbor in adjacency[atom_index]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        if (len(component) >= min_chain_length
                and all(len(adjacency[index]) <= 2 for index in component)):
            chains.append(component)
    return chains


def coarse_grained_smiles2graph(mol, min_chain_length):
    """Merge eligible carbon chains into pseudo nodes with a chain-length field."""
    if min_chain_length < 2:
        raise ValueError('min_chain_length must be at least 2.')

    heavy_mol = Chem.Mol(mol)
    chains = find_linear_carbon_chains(heavy_mol, min_chain_length)
    if not chains:
        graph = smiles2graph(mol, coarse_grain=False)
        graph['node_feat'] = np.column_stack([
            graph['node_feat'],
            np.zeros(graph['num_nodes'], dtype=np.int64),
        ])
        return graph

    molecule = Chem.AddHs(heavy_mol)
    collapsed_atoms = {index for chain in chains for index in chain}
    removed_hydrogens = {
        atom.GetIdx() for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 1
        and any(neighbor.GetIdx() in collapsed_atoms
                for neighbor in atom.GetNeighbors())
    }
    retained_atoms = [
        atom.GetIdx() for atom in molecule.GetAtoms()
        if atom.GetIdx() not in collapsed_atoms
        and atom.GetIdx() not in removed_hydrogens
    ]

    node_features = []
    node_mapping = {}
    for atom_index in retained_atoms:
        node_mapping[atom_index] = len(node_features)
        node_features.append(atom_to_feature_vector(
            molecule.GetAtomWithIdx(atom_index)
        ) + [0])

    for chain in chains:
        pseudo_index = len(node_features)
        for atom_index in chain:
            node_mapping[atom_index] = pseudo_index
        pseudo_feature = atom_to_feature_vector(
            heavy_mol.GetAtomWithIdx(chain[0])
        )
        pseudo_feature[0] = safe_index(
            allowable_features['possible_atomic_num_list'], 'misc'
        )
        pseudo_feature[2] = safe_index(
            allowable_features['possible_degree_list'], 0
        )
        node_features.append(pseudo_feature + [len(chain)])

    edges_list, edge_features_list, seen_edges = [], [], set()
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin not in node_mapping or end not in node_mapping:
            continue
        mapped_begin, mapped_end = node_mapping[begin], node_mapping[end]
        if mapped_begin == mapped_end:
            continue
        edge_key = tuple(sorted((mapped_begin, mapped_end)))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edge_feature = bond_to_feature_vector(bond)
        edges_list.extend([(mapped_begin, mapped_end), (mapped_end, mapped_begin)])
        edge_features_list.extend([edge_feature, edge_feature])

    edge_index = (np.array(edges_list, dtype=np.int64).T if edges_list
                  else np.empty((2, 0), dtype=np.int64))
    edge_feat = (np.array(edge_features_list, dtype=np.int64) if edge_features_list
                 else np.empty((0, 1), dtype=np.int64))
    return {
        'edge_index': edge_index,
        'edge_feat': edge_feat,
        'node_feat': np.array(node_features, dtype=np.int64),
        'num_nodes': len(node_features),
    }

# allowable multiple choice node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1
# # miscellaneous case
# i = safe_index(allowable_features['possible_atomic_num_list'], 'asdf')
# assert allowable_features['possible_atomic_num_list'][i] == 'misc'
# # normal case
# i = safe_index(allowable_features['possible_atomic_num_list'], 2)
# assert allowable_features['possible_atomic_num_list'][i] == 2

# def atom_to_feature_vector(atom):
#     """
#     Converts rdkit atom object to feature list of indices
#     :param mol: rdkit atom object
#     :return: list
#     """
#     #### original version
#     atom_feature = [
#             safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
#             allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
#             safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
#             safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
#             safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
#             safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
#             safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
#             allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
#             allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
#             ]
#     return atom_feature
# from rdkit import Chem
# mol = Chem.MolFromSmiles('Cl[C@H](/C=C/C)Br')
# atom = mol.GetAtomWithIdx(1)  # chiral carbon
# atom_feature = atom_to_feature_vector(atom)
# assert atom_feature == [5, 2, 4, 5, 1, 0, 2, 0, 0]

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    #### modified version
    atom_feature = [
        safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
        allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
        safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
        safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
        safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
        safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
        safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
        allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
    ]

    # Check if the atomic number is "*", if so, set it to the special index
    if atom.GetSymbol() == '*':
        atom_feature[0] = safe_index(allowable_features['possible_atomic_num_list'], 'misc')

    return atom_feature

def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
        ]))


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    #### modified version
    bond_feature = [
        safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType()))
    ]

    # Check if the bond type is "*", if so, set it to the special index
    if bond.GetBondType() == Chem.rdchem.BondType.UNSPECIFIED:
        bond_feature[0] = safe_index(allowable_features['possible_bond_type_list'], 'misc')

    return bond_feature
# def bond_to_feature_vector(bond):
#     """
#     Converts rdkit bond object to feature list of indices
#     :param mol: rdkit bond object
#     :return: list
#     """
#     #### original version
#     # bond_feature = [
#     #             safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
#     #             allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
#     #             allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
#     #         ]
#     #### defined version
#     bond_feature = [
#         safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType()))
#     ]
#
#     return bond_feature
# uses same molecule as atom_to_feature_vector test
# bond = mol.GetBondWithIdx(2)  # double bond with stereochem
# bond_feature = bond_to_feature_vector(bond)
# assert bond_feature == [1, 2, 0]

def get_bond_feature_dims():
    ### defined version
    return list(map(len,
        allowable_features['possible_bond_type_list']
        ))
    ## orginal version
    # return list(map(len, [
    #     allowable_features['possible_bond_type_list'],
    #     allowable_features['possible_bond_stereo_list'],
    #     allowable_features['possible_is_conjugated_list']
    # ]))

def atom_feature_vector_to_dict(atom_feature):
    [atomic_num_idx,
    chirality_idx,
    degree_idx,
    formal_charge_idx,
    num_h_idx,
    number_radical_e_idx,
    hybridization_idx,
    is_aromatic_idx,
    is_in_ring_idx] = atom_feature

    feature_dict = {
        'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
        'chirality': allowable_features['possible_chirality_list'][chirality_idx],
        'degree': allowable_features['possible_degree_list'][degree_idx],
        'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx],
        'num_h': allowable_features['possible_numH_list'][num_h_idx],
        'num_rad_e': allowable_features['possible_number_radical_e_list'][number_radical_e_idx],
        'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
        'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
        'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
    }

    return feature_dict
# # uses same atom_feature as atom_to_feature_vector test
# atom_feature_dict = atom_feature_vector_to_dict(atom_feature)
# assert atom_feature_dict['atomic_num'] == 6
# assert atom_feature_dict['chirality'] == 'CHI_TETRAHEDRAL_CCW'
# assert atom_feature_dict['degree'] == 4
# assert atom_feature_dict['formal_charge'] == 0
# assert atom_feature_dict['num_h'] == 1
# assert atom_feature_dict['num_rad_e'] == 0
# assert atom_feature_dict['hybridization'] == 'SP3'
# assert atom_feature_dict['is_aromatic'] == False
# assert atom_feature_dict['is_in_ring'] == False

def bond_feature_vector_to_dict(bond_feature):
    ### define version
    [bond_type_idx] = bond_feature

    feature_dict = {
        'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx]
    }
    #### original version
    # [bond_type_idx,
    #  bond_stereo_idx,
    #  is_conjugated_idx] = bond_feature
    #
    # feature_dict = {
    #     'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
    #     'bond_stereo': allowable_features['possible_bond_stereo_list'][bond_stereo_idx],
    #     'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx]
    # }

    return feature_dict
# # uses same bond as bond_to_feature_vector test
# bond_feature_dict = bond_feature_vector_to_dict(bond_feature)
# assert bond_feature_dict['bond_type'] == 'DOUBLE'
# assert bond_feature_dict['bond_stereo'] == 'STEREOE'
# assert bond_feature_dict['is_conjugated'] == False
if __name__ == '__main__':
    graph = smiles2graph('O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5')
    print(graph)

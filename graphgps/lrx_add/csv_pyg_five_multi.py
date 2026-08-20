"""
@Name:  csv_pyg.py
@Auth:  rongxing
@Date:  2023/1/9-下午9:11
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME}
from ogb.utils import smiles2graph
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
bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType()))
            ]
2023.04.10 train use repeat_number_train
valid and test use repeat_number_test_valid
multiple smiles own error between -2.5 and 2.5 temperature  for train datasets
"""
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data as PyGData
import numpy as np
from scipy.sparse import coo_matrix
import os
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdFingerprintGenerator
from sklearn.model_selection import train_test_split
import os.path as osp
from torch_geometric.data import InMemoryDataset
###from ogb.utils import smiles2graph  original version
from ogb.utils.torch_util import replace_numpy_with_torchtensor
import random
from graph_feature import smiles2graph
from graphgps.lrx_add.mordred_lookup import mordred_feature_vector
from graphgps.lrx_add.fifth_descriptor_lookup import fifth_descriptor_vector
from graphgps.lrx_add.fifth_semantic_lookup import fifth_semantic_vector
from graphgps.lrx_add.fifth_structured_lookup import fifth_structured_values
from graphgps.component_aux import component_aux_enabled

from torch_geometric.utils.convert import to_networkx
import networkx as nx
from torch_geometric.graphgym.config import cfg
# from ScaffoldSplitter import ScaffoldSplitter,RandomScaffoldSplitter
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions


MOLECULAR_AUX_DIM = 136
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=128
)
_CURRENT_MORDRED_FEATURE = None
_CURRENT_FIFTH_MECHANISTIC_FEATURE = None
_CURRENT_FIFTH_SEMANTIC_FEATURE = None
_CURRENT_FIFTH_AA_ID = None
_CURRENT_FIFTH_TERMINAL_ID = None
_CURRENT_FIFTH_TAIL = None
_CURRENT_SAMPLE_UID = None
_CURRENT_COMPONENT_VOCAB_ID = None
_CURRENT_FIFTH_CLASS_ID = None


def Data(*args, **kwargs):
    kwargs.setdefault('mordred_feat', _CURRENT_MORDRED_FEATURE)
    kwargs.setdefault('fifth_mechanistic_feat', _CURRENT_FIFTH_MECHANISTIC_FEATURE)
    kwargs.setdefault('fifth_semantic_feat', _CURRENT_FIFTH_SEMANTIC_FEATURE)
    kwargs.setdefault('fifth_aa_id', _CURRENT_FIFTH_AA_ID)
    kwargs.setdefault('fifth_terminal_id', _CURRENT_FIFTH_TERMINAL_ID)
    kwargs.setdefault('fifth_tail_feat', _CURRENT_FIFTH_TAIL)
    kwargs.setdefault('sample_uid', _CURRENT_SAMPLE_UID)
    kwargs.setdefault('component_vocab_id', _CURRENT_COMPONENT_VOCAB_ID)
    kwargs.setdefault('fifth_class_id', _CURRENT_FIFTH_CLASS_ID)
    return PyGData(*args, **kwargs)


def canonical_component_smiles(smiles):
    """Return a stable SMILES key without looking outside the current CSV."""
    if str(smiles) in {'nan', '[Fr]'}:
        return '[Fr]'
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return '[Fr]'
    return Chem.MolToSmiles(molecule, canonical=True)


def build_input_component_vocab(data, reserve_unknown=True):
    """Build five input-only component vocabularies from the active CSV.

    ``reserve_unknown=False`` is a strict ablation for the first four
    component positions: only canonical structures present in ``data`` are
    assigned IDs, starting at zero.  Missing or malformed first-four
    structures are rejected instead of receiving an unknown category.  The
    fifth position can legitimately be absent and is graph encoded by O12, so
    an observed ``[Fr]`` remains part of its source vocabulary.
    """
    columns = ['IL_SMILE', 'HL_SMILE', 'Chol_SMILE', 'PEG_SMILE', 'Fifth_SMILE']
    vocabularies = []
    for component_index, column in enumerate(columns):
        keys = sorted({canonical_component_smiles(value) for value in data[column]})
        if not reserve_unknown and component_index < 4 and '[Fr]' in keys:
            raise ValueError(
                f'Strict component vocabulary rejects missing or malformed '
                f'{column} values in the vocabulary source.')
        # The normal path reserves a deterministic unknown/placeholder item.
        # The strict path contains only source-data categories.
        if reserve_unknown and '[Fr]' not in keys:
            keys.insert(0, '[Fr]')
        vocabularies.append({key: index for index, key in enumerate(keys)})
    return vocabularies


def component_vocab_id(vocabulary, smiles, component_index, strict=False):
    """Resolve one component ID and validate it before embedding lookup."""
    component_key = canonical_component_smiles(smiles)
    if strict and component_key not in vocabulary:
        raise ValueError(
            f'Strict component vocabulary has no entry for '
            f'component {component_index}: {component_key!r}.')
    vocab_id = vocabulary.get(component_key, 0)
    if vocab_id < 0 or vocab_id >= len(vocabulary):
        raise IndexError(
            f'Component {component_index} vocabulary ID {vocab_id} is '
            f'outside [0, {len(vocabulary) - 1}].')
    return vocab_id


def canonical_fifth_class(value):
    """Normalize a user-supplied fifth-component class without target data."""
    if pd.isna(value) or str(value).strip() == '':
        return '__unknown__'
    return str(value).strip().lower()


def build_input_fifth_class_vocab(data):
    """Build the deterministic fifth-class vocabulary from the input CSV."""
    values = (
        data['Fifth_class']
        if 'Fifth_class' in data.columns
        else pd.Series(['__unknown__'])
    )
    keys = sorted({canonical_fifth_class(value) for value in values})
    if '__unknown__' not in keys:
        keys.insert(0, '__unknown__')
    return {key: index for index, key in enumerate(keys)}


def input_fifth_class_id(vocabulary, value):
    """Map a class label to the input-derived vocabulary's unknown-safe ID."""
    return vocabulary.get(
        canonical_fifth_class(value), vocabulary['__unknown__'])


def molecular_aux_features(molecule):
    """Return 128 Morgan bits and eight bounded RDKit descriptors."""
    fingerprint = np.zeros(128, dtype=np.float32)
    if molecule is None:
        return np.zeros(MOLECULAR_AUX_DIM, dtype=np.float32)
    bit_vector = MORGAN_GENERATOR.GetFingerprint(molecule)
    DataStructs.ConvertToNumpyArray(bit_vector, fingerprint)
    descriptors = np.array([
        Descriptors.MolWt(molecule) / 1000.0,
        Descriptors.MolLogP(molecule) / 10.0,
        Descriptors.TPSA(molecule) / 200.0,
        Lipinski.NumHDonors(molecule) / 10.0,
        Lipinski.NumHAcceptors(molecule) / 20.0,
        Lipinski.NumRotatableBonds(molecule) / 20.0,
        Lipinski.RingCount(molecule) / 10.0,
        Lipinski.FractionCSP3(molecule),
    ], dtype=np.float32)
    return np.concatenate([fingerprint, descriptors])




def similar_(mol):

    target_smi ='O=C1CCC(=O)N1C' # MI_V1
    # target_smi = 'O=C1C=CC(=O)N1C' # MI_V2
    # target_smi = 'O=C1C=CC(=O)N1CN2C(=O)C=CC2=O' # BMI
    target_mol = Chem.MolFromSmiles(target_smi)

    target_fp_RDKFingerprint = Chem.RDKFingerprint(target_mol)
    fp_RDKFingerprint = Chem.RDKFingerprint(mol)
    # print('RDKFingerprint', DataStructs.FingerprintSimilarity(fp_RDKFingerprint, target_fp_RDKFingerprint))
    similar_value = DataStructs.FingerprintSimilarity(fp_RDKFingerprint, target_fp_RDKFingerprint)
    return similar_value

def tg_weigth_(tg):
    alpha = 0.1
    weights =0
    critical_value = 300

    ##### version 1
    if tg >=critical_value:
        weights = 1+ round((alpha * (tg - critical_value)),2) #round(np.exp(alpha * (tg - 200)),2)  # 对于T >= 200，权重指数增长
        # weights = max((1 + round((alpha * (tg - critical_value)), 2)),10)
    else:
        weights = round(np.exp(-alpha * (critical_value - tg)),4) # 对于T < 200，权重指数衰减  <1 = 0.05
    # print(tg,weights)
    return weights

def addCyclicConnection(mol):
    stars = []
    nbs = []
    # mol = Chem.MolFromSmiles('CCCCCCC1(CCCCCC)c2cc(*)ccc2-c2c1cc(cc2)c1ccc2-c3c([C]4(=CC=[C](C=C4)(C4=NC(CO4)c4ccccc4)C4=NC(CO4)c4ccccc4)c2c1)cc(cc3)*')
    for j, atom in enumerate(mol.GetAtoms()):
        atom_symbol = atom.GetSymbol()
        if atom_symbol == '*':
            bonds = list(atom.GetBonds())
            assert len(bonds) == 1
            stars.append(atom.GetIdx())
            bond_type = bonds[0].GetBondType()
            for a in atom.GetNeighbors():
                nbs.append(a.GetIdx())
    edmol = Chem.EditableMol(mol)
    # Draw.MolToFile(mol,f'mol_{random.randrange(0, 100000000)}.png')
    try:
        edmol.AddBond(nbs[0], nbs[1], order=bond_type)
    except RuntimeError:
        print('') #print('bond already exists, skipping...')
    if (stars[0] > stars[1]):
        edmol.RemoveAtom(stars[0])
        edmol.RemoveAtom(stars[1])
    else:
        edmol.RemoveAtom(stars[1])
        edmol.RemoveAtom(stars[0])
    return edmol.GetMol()

def smiles_to_data_5(smiles_item, label_pair_list,ratio_pair_list,
                     data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5,property_num,property_name,
                     sample_index=None, component_vocabularies=None,
                     fifth_class_id=None):
    global _CURRENT_MORDRED_FEATURE, _CURRENT_FIFTH_MECHANISTIC_FEATURE, _CURRENT_FIFTH_SEMANTIC_FEATURE, _CURRENT_FIFTH_AA_ID, _CURRENT_FIFTH_TERMINAL_ID, _CURRENT_FIFTH_TAIL
    global _CURRENT_SAMPLE_UID, _CURRENT_COMPONENT_VOCAB_ID
    global _CURRENT_FIFTH_CLASS_ID
    _CURRENT_SAMPLE_UID = torch.tensor([int(sample_index)], dtype=torch.long) if sample_index is not None else None
    _CURRENT_FIFTH_CLASS_ID = torch.tensor(
        [int(fifth_class_id or 0)], dtype=torch.long)
    for num, smiles_item_each in enumerate(smiles_item):
        # print('serial-num= ', num)
        try:
            if component_vocabularies is not None:
                vocabulary = component_vocabularies[num]
                vocab_id = component_vocab_id(
                    vocabulary,
                    smiles_item_each,
                    num + 1,
                    strict=bool(getattr(cfg, 'component_vocab_strict', False)),
                )
                _CURRENT_COMPONENT_VOCAB_ID = torch.tensor([vocab_id], dtype=torch.long)
            else:
                _CURRENT_COMPONENT_VOCAB_ID = torch.tensor([0], dtype=torch.long)
            if str(smiles_item_each) != 'nan' and str(smiles_item_each) != '[Fr]':
                mol_ = Chem.MolFromSmiles(smiles_item_each)
                graph = smiles2graph(
                    mol_, cfg.coarse_grain_enable,
                    cfg.coarse_grain_min_chain_length,
                )
                mask_ =False
            else:
                smiles_item_each = '[Fr]'
                mol_ = Chem.MolFromSmiles(smiles_item_each)
                graph = smiles2graph(
                    mol_, cfg.coarse_grain_enable,
                    cfg.coarse_grain_min_chain_length,
                )
                mask_ = True

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            x = torch.from_numpy(graph['node_feat']).to(torch.int64)

            edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            edge_attr = torch.from_numpy(graph['edge_feat'].flatten()).to(torch.long)
            if component_aux_enabled(cfg, num):
                aux_feat = torch.from_numpy(
                    molecular_aux_features(mol_)
                ).view(1, -1)
            else:
                aux_feat = torch.zeros((1, MOLECULAR_AUX_DIM), dtype=torch.float32)
            _CURRENT_MORDRED_FEATURE = torch.from_numpy(mordred_feature_vector(
                smiles_item_each, cfg.use_mordred_features,
                cfg.mordred_feature_path, cfg.mordred_feature_dim,
            )).view(1, -1)
            # O13-E descriptors belong only to component 5. Components 1-4
            # receive a same-shaped zero tensor so PyG batching is stable,
            # while the model consumes only ``data5`` below.
            _CURRENT_FIFTH_MECHANISTIC_FEATURE = torch.from_numpy(
                fifth_descriptor_vector(
                    smiles_item_each,
                    bool(cfg.use_fifth_mechanistic_descriptors and num == 4),
                    cfg.fifth_mechanistic_descriptor_path,
                    cfg.fifth_mechanistic_descriptor_dim,
                )
            ).view(1, -1)
            _CURRENT_FIFTH_SEMANTIC_FEATURE = torch.from_numpy(
                fifth_semantic_vector(
                    smiles_item_each,
                    bool(cfg.use_fifth_semantic_features and num == 4),
                    cfg.fifth_semantic_feature_path,
                    cfg.fifth_semantic_feature_dim,
                )
            ).view(1, -1)
            aa_id, terminal_id, tail, tail_mask = fifth_structured_values(
                smiles_item_each, bool(cfg.use_fifth_structured_features and num == 4),
                cfg.fifth_structured_feature_path)
            _CURRENT_FIFTH_AA_ID = torch.tensor([aa_id], dtype=torch.long)
            _CURRENT_FIFTH_TERMINAL_ID = torch.tensor([terminal_id], dtype=torch.long)
            _CURRENT_FIFTH_TAIL = torch.tensor([[tail, tail_mask]], dtype=torch.float32)

            if property_num == 1:
                target_index = int(getattr(cfg, 'single_task_target_index', property_name))
                if target_index not in range(6):
                    raise ValueError(
                        'single_task_target_index must be an integer from 0 to 5, '
                        f'got {target_index}.')
                # The first four efficiency labels use the historical /100
                # normalization; Norm_before and Norm_after remain in their
                # original units, exactly as in the two-task loader branch.
                target_value = label_pair_list[target_index]
                y = torch.Tensor([target_value / 100 if target_index < 4 else target_value])
                if num == 0:
                    data_1 = Data(x, edge_index, edge_attr, y, ratio=ratio_pair_list[num], mask=mask_, aux_feat=aux_feat)
                    data_sum_1.append(data_1)
                elif num == 1:
                    data_2 = Data(x, edge_index, edge_attr, y, ratio=ratio_pair_list[num], mask=mask_, aux_feat=aux_feat)
                    data_sum_2.append(data_2)
                elif num == 2:
                    data_3 = Data(x, edge_index, edge_attr, y, ratio=ratio_pair_list[num], mask=mask_, aux_feat=aux_feat)
                    data_sum_3.append(data_3)
                elif num == 3:
                    data_4 = Data(x, edge_index, edge_attr, y, ratio=ratio_pair_list[num], mask=mask_, aux_feat=aux_feat)
                    data_sum_4.append(data_4)
                elif num == 4:
                    if str(smiles_item_each) == 'nan' or str(ratio_pair_list[num]) == 'nan':
                        data_5 = Data(x, edge_index, edge_attr, y, ratio=0.0, mask=mask_, aux_feat=aux_feat)
                    else:
                        data_5 = Data(x, edge_index, edge_attr, y, ratio=ratio_pair_list[num], mask=mask_, aux_feat=aux_feat)
                    data_sum_5.append(data_5)
            elif property_num == 4:
                # A four-task model can select any four source labels.  The
                # historical core4 setup remains the default; later4 selects
                # [Aerosolization, Recovery, Norm_before, Norm_after].
                target_indices = list(getattr(
                    cfg, 'multi_task_target_indices', [0, 1, 2, 3]))
                if len(target_indices) != 4 or any(index not in range(6) for index in target_indices):
                    raise ValueError(
                        'multi_task_target_indices must contain four source-label indices in [0, 5].')
                selected_labels = [
                    label_pair_list[index] / 100 if index < 4 else label_pair_list[index]
                    for index in target_indices
                ]
                y = torch.Tensor([selected_labels[0]])
                y1 = torch.Tensor([selected_labels[1]])
                y2 = torch.Tensor([selected_labels[2]])
                y3 = torch.Tensor([selected_labels[3]])
                if num == 0:
                    data_1 = Data(x, edge_index, edge_attr, y, y1=y1, y2=y2, y3=y3, ratio=ratio_pair_list[num], mask=mask_, aux_feat=aux_feat)
                    data_sum_1.append(data_1)
                elif num == 1:
                    data_2 = Data(x, edge_index, edge_attr, y, y1=y1, y2=y2, y3=y3, ratio=ratio_pair_list[num], mask=mask_, aux_feat=aux_feat)
                    data_sum_2.append(data_2)
                elif num == 2:
                    data_3 = Data(x, edge_index, edge_attr, y, y1=y1, y2=y2, y3=y3, ratio=ratio_pair_list[num], mask=mask_, aux_feat=aux_feat)
                    data_sum_3.append(data_3)
                elif num == 3:
                    data_4 = Data(x, edge_index, edge_attr, y, y1=y1, y2=y2, y3=y3, ratio=ratio_pair_list[num], mask=mask_, aux_feat=aux_feat)
                    data_sum_4.append(data_4)
                elif num == 4:
                    if str(smiles_item_each) == 'nan' or str(ratio_pair_list[num]) == 'nan' \
                            or float(ratio_pair_list[num]) <= 1e-10:
                        data_5 = Data(x, edge_index, edge_attr, y, y1=y1, y2=y2, y3=y3, ratio=0.0, mask=mask_, aux_feat=aux_feat)
                        data_sum_5.append(data_5)
                    else:
                        data_5 = Data(x, edge_index, edge_attr, y, y1=y1, y2=y2, y3=y3, ratio=ratio_pair_list[num],
                                      mask=mask_, aux_feat=aux_feat)
                        data_sum_5.append(data_5)
            elif property_num == 2:
                y = torch.Tensor([label_pair_list[4]])
                y1 = torch.Tensor([label_pair_list[5]])
                if num == 0:
                    data_1 = Data(x, edge_index, edge_attr, y, y1=y1, ratio=ratio_pair_list[num], mask=mask_, aux_feat=aux_feat)
                    data_sum_1.append(data_1)
                elif num == 1:
                    data_2 = Data(x, edge_index, edge_attr, y, y1=y1, ratio=ratio_pair_list[num], mask=mask_, aux_feat=aux_feat)
                    data_sum_2.append(data_2)
                elif num == 2:
                    data_3 = Data(x, edge_index, edge_attr, y, y1=y1, ratio=ratio_pair_list[num], mask=mask_, aux_feat=aux_feat)
                    data_sum_3.append(data_3)
                elif num == 3:
                    data_4 = Data(x, edge_index, edge_attr, y, y1=y1, ratio=ratio_pair_list[num], mask=mask_, aux_feat=aux_feat)
                    data_sum_4.append(data_4)
                elif num == 4:
                    if str(smiles_item_each) == 'nan' or str(ratio_pair_list[num]) == 'nan':
                        data_5 = Data(x, edge_index, edge_attr, y, y1=y1, ratio=0.0, mask=mask_, aux_feat=aux_feat)
                        data_sum_5.append(data_5)
                    else:
                        data_5 = Data(x, edge_index, edge_attr, y, y1=y1, ratio=ratio_pair_list[num],
                                      mask=mask_, aux_feat=aux_feat)
                        data_sum_5.append(data_5)

        except:
            print('graph generate error:', num, smiles_item_each,smiles_item, y)
            # break

    return data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5


class LRX_five_multi(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,subset: bool = False,
        split: str = 'train',pre_filter= None):
        self.subset = subset
        # assert split in ['train', 'val', 'test','train_j', 'val_j','test_j']
        assert split in ['train','val', 'test',
                         'train_2', 'val_2','test_2',
                         'train_3','val_3', 'test_3',
                         'train_4', 'val_4','test_4',
                         'train_5','val_5','test_5']
        # assert split in ['train','train_2','train_3','train_4','train_5',
        #                  'val','val_2','val_3','val_4','val_5',
        #                  'test', 'test_2','test_3','test_4','test_5']
        super(LRX_five_multi, self).__init__(root, transform, pre_transform,pre_filter)
        self.data, self.slices = torch.load(
            osp.join(self.processed_dir, f'{split}.pt'), weights_only=False)


    @property
    def raw_file_names(self) :
        return cfg.read_csv

    @property
    def processed_dir(self) -> str:
        name = 'subset' if self.subset else 'full'
        # print('processed_dir',osp.join(self.root, name, 'processed'))
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self) :
        return ['train.pt', 'val.pt','test.pt',
                'train_2.pt', 'val_2.pt','test_2.pt',
                'train_3.pt', 'val_3.pt', 'test_3.pt',
                'train_4.pt', 'val_4.pt','test_4.pt',
                'train_5.pt','val_5.pt','test_5.pt']

    def download(self):
        pass

    def process(self):
        ####
        property_num = cfg.property_num
        property_name = cfg.property_serial

        ####
        data = pd.read_csv(self.raw_paths[0])
        data['_stage3_source_index'] = np.arange(len(data), dtype=np.int64)
        # A trained OneHotEmbedGPS checkpoint owns the categorical vocabulary
        # used to size and index its first-four-component embeddings.  During
        # external inference, derive that vocabulary from the original input
        # CSV when cfg.component_vocab_source is supplied; unseen external
        # molecules then map to the established unknown ID instead of
        # rebuilding (and silently reindexing) the checkpoint vocabulary.
        vocabulary_source = str(getattr(cfg, 'component_vocab_source', '')).strip()
        source_path = os.path.abspath(self.raw_paths[0])
        if vocabulary_source and os.path.isfile(vocabulary_source):
            vocabulary_path = os.path.abspath(vocabulary_source)
            vocabulary_data = data if vocabulary_path == source_path else pd.read_csv(vocabulary_path)
        else:
            vocabulary_path = source_path
            vocabulary_data = data
        component_vocabularies = build_input_component_vocab(
            vocabulary_data,
            reserve_unknown=not bool(getattr(cfg, 'component_vocab_strict', False)),
        )
        fifth_class_vocabulary = build_input_fifth_class_vocab(vocabulary_data)
        cfg.component_vocab_sizes = [len(vocabulary) for vocabulary in component_vocabularies[:4]]
        cfg.fifth_component_vocab_size = len(component_vocabularies[4])
        cfg.fifth_class_vocab_size = len(fifth_class_vocabulary)
        cfg.component_vocab_source = vocabulary_path
        diagnostic_split_path = str(cfg.dataset.diagnostic_split_path).strip()
        if diagnostic_split_path:
            split_data = pd.read_csv(diagnostic_split_path)
            id_column = str(cfg.dataset.diagnostic_id_column)
            manifest_id_column = str(cfg.dataset.diagnostic_manifest_id_column).strip() or id_column
            required_columns = {manifest_id_column, 'split'}
            if not required_columns.issubset(split_data.columns):
                raise ValueError(
                    f'Diagnostic split {diagnostic_split_path} must contain '
                    f'{sorted(required_columns)}.'
                )
            if data[id_column].duplicated().any() or split_data[manifest_id_column].duplicated().any():
                raise ValueError(
                    'Diagnostic split requires unique source and manifest identifiers '
                    f'({id_column}, {manifest_id_column}).'
                )
            split_columns = [manifest_id_column, 'split']
            if 'split_order' in split_data.columns:
                split_columns.append('split_order')
            split_data = split_data[split_columns].copy()
            split_data[manifest_id_column] = split_data[manifest_id_column].astype(str)
            if 'split_order' in split_data.columns:
                split_data = split_data.sort_values(['split', 'split_order'], kind='stable')
            indexed_data = data.copy()
            indexed_data[id_column] = indexed_data[id_column].astype(str)
            indexed_data = indexed_data.set_index(id_column, drop=False)
            missing_ids = set(split_data[manifest_id_column]) - set(indexed_data.index)
            if missing_ids:
                raise ValueError(
                    f'Diagnostic split contains IDs absent from training CSV: '
                    f'{sorted(missing_ids)[:5]}'
                )
            if len(split_data) != len(indexed_data):
                raise ValueError(
                    'Diagnostic split must cover every row in the training CSV exactly once.'
                )
            train_data = indexed_data.loc[
                split_data.loc[split_data['split'] == 'train', manifest_id_column]
            ].reset_index(drop=True)
            valid_data = indexed_data.loc[
                split_data.loc[split_data['split'] == 'val', manifest_id_column]
            ].reset_index(drop=True)
            test_data = indexed_data.loc[
                split_data.loc[split_data['split'] == 'test', manifest_id_column]
            ].reset_index(drop=True)
            if not all((len(train_data), len(valid_data), len(test_data))):
                raise ValueError('Diagnostic split has an empty train, val, or test partition.')
            print(f'Using explicit diagnostic split: {diagnostic_split_path}')
        else:
            train_data, test_data = train_test_split(data, train_size=0.9, test_size=0.1, random_state = int(cfg.seed))
            train_data, valid_data = train_test_split(train_data, train_size=0.9, test_size=0.1, random_state = int(cfg.seed))
        print(len(train_data),len(test_data),len(valid_data))

        list_splite = [train_data, valid_data, test_data]
        for key,item in enumerate(list_splite):
            # item = item.reset_index(drop=True)
            idx_sum = len(item)
            print('key =', key, 'len(item)',idx_sum)
            # Tm_list = list(item['Tg_mean'])
            molecule_1 = list(item['IL_SMILE'])
            molecule_2 = list(item['HL_SMILE'])
            molecule_3 = list(item['Chol_SMILE'])
            molecule_4 = list(item['PEG_SMILE'])
            molecule_5 = list(item['Fifth_SMILE'])
            rate_1 = list(item['mol%_IL']*0.01)
            rate_2 = list(item['mol%_HL']*0.01)
            rate_3 = list(item['mol%_Chol']*0.01)
            rate_4 = list(item['mol%_PEG']*0.01)
            rate_5 = list(item['mol%_Fifth']*0.01)
            y1_list = list(item['EE_before']) #item['EE_before'].tolist()
            y2_list = list(item['EE_after']) #item['EE_after'].tolist()
            y3_list = list(item['Aerosolization_Efficiency']) #item['Aerosolization_Efficiency'].tolist()
            y4_list = list(item['mRNA_Recovery_Efficiency']) #item['mRNA_Recovery_Efficiency'].tolist()
            y5_list = list(item['Norm_before']) #item['Norm_before'].tolist()
            y6_list = list(item['Norm_after']) #item['Norm_after'].tolist()


            smiles_list = [list(pair_smi) for pair_smi in zip(molecule_1, molecule_2,molecule_3,molecule_4,molecule_5)]
            ratio_list = [list(pair_rate) for pair_rate in zip(rate_1, rate_2, rate_3, rate_4, rate_5)]
            label_list = [list(pair_y) for pair_y in zip(y1_list, y2_list, y3_list, y4_list, y5_list,y6_list)]
            fifth_class_list = (
                list(item['Fifth_class'])
                if 'Fifth_class' in item.columns
                else ['__unknown__'] * len(item)
            )
            sample_index_list = list(item['_stage3_source_index'])

            data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5 = [],[],[],[],[]

            if key == 0 :
                for idx in range(idx_sum):
                    if (idx %2000) == 0 :
                        print('idx',idx)
                    smi_pair_list, label_pair_list, ratio_pair_list = smiles_list[idx], label_list[idx], ratio_list[idx]
                    fifth_class_id = input_fifth_class_id(
                        fifth_class_vocabulary, fifth_class_list[idx])
                    data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5 = smiles_to_data_5(smi_pair_list, label_pair_list,ratio_pair_list,
                                                              data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5,property_num,property_name,
                                                              sample_index=sample_index_list[idx], component_vocabularies=component_vocabularies,
                                                              fifth_class_id=fifth_class_id)
            else:
                for idx in range(idx_sum):
                    if (idx %2000) == 0 :
                        print('idx',idx)
                    smi_pair_list, label_pair_list, ratio_pair_list = smiles_list[idx], label_list[idx], ratio_list[idx]
                    fifth_class_id = input_fifth_class_id(
                        fifth_class_vocabulary, fifth_class_list[idx])
                    data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5 = smiles_to_data_5(smi_pair_list, label_pair_list,ratio_pair_list,
                                                              data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5,property_num,property_name,
                                                              sample_index=sample_index_list[idx], component_vocabularies=component_vocabularies,
                                                              fifth_class_id=fifth_class_id)


            if key == 0:
                ##打乱顺序
                # random.shuffle(data_sum)
                print('train :len(data_sum_1-5',
                      len(data_sum_1), len(data_sum_2), len(data_sum_3), len(data_sum_4), len(data_sum_5))
                torch.save(self.collate(data_sum_1), os.path.join(self.processed_dir, f'train.pt'))
                torch.save(self.collate(data_sum_2), os.path.join(self.processed_dir, f'train_2.pt'))
                torch.save(self.collate(data_sum_3), os.path.join(self.processed_dir, f'train_3.pt'))
                torch.save(self.collate(data_sum_4), os.path.join(self.processed_dir, f'train_4.pt'))
                torch.save(self.collate(data_sum_5), os.path.join(self.processed_dir, f'train_5.pt'))

            elif key == 1 :
                print('val :len(data_sum_1-5',
                      len(data_sum_1), len(data_sum_2), len(data_sum_3), len(data_sum_4), len(data_sum_5))
                torch.save(self.collate(data_sum_1), os.path.join(self.processed_dir, f'val.pt'))
                torch.save(self.collate(data_sum_2), os.path.join(self.processed_dir, f'val_2.pt'))
                torch.save(self.collate(data_sum_3), os.path.join(self.processed_dir, f'val_3.pt'))
                torch.save(self.collate(data_sum_4), os.path.join(self.processed_dir, f'val_4.pt'))
                torch.save(self.collate(data_sum_5), os.path.join(self.processed_dir, f'val_5.pt'))

            elif key == 2:
                print('test :len(data_sum_1-5',
                      len(data_sum_1), len(data_sum_2), len(data_sum_3), len(data_sum_4), len(data_sum_5))
                torch.save(self.collate(data_sum_1), os.path.join(self.processed_dir, f'test.pt'))
                torch.save(self.collate(data_sum_2), os.path.join(self.processed_dir, f'test_2.pt'))
                torch.save(self.collate(data_sum_3), os.path.join(self.processed_dir, f'test_3.pt'))
                torch.save(self.collate(data_sum_4), os.path.join(self.processed_dir, f'test_4.pt'))
                torch.save(self.collate(data_sum_5), os.path.join(self.processed_dir, f'test_5.pt'))
                # print('test :len(data_sum_1),len(data_sum_2)', len(data_sum_i), len(data_sum_j))
                # torch.save(self.collate(data_sum_i), os.path.join(self.processed_dir, f'test.pt'))
                # torch.save(self.collate(data_sum_j), os.path.join(self.processed_dir, f'test_j.pt'))
            print('------------ending--------------')

        return data_sum_1, data_sum_2, data_sum_3, data_sum_4, data_sum_5

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
_CURRENT_SAMPLE_UID = None
_CURRENT_COMPONENT_VOCAB_ID = None


def Data(*args, **kwargs):
    kwargs.setdefault('mordred_feat', _CURRENT_MORDRED_FEATURE)
    kwargs.setdefault('sample_uid', _CURRENT_SAMPLE_UID)
    kwargs.setdefault('component_vocab_id', _CURRENT_COMPONENT_VOCAB_ID)
    return PyGData(*args, **kwargs)


def canonical_component_smiles(smiles):
    """Return a stable SMILES key without looking outside the current CSV."""
    if str(smiles) in {'nan', '[Fr]'}:
        return '[Fr]'
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return '[Fr]'
    return Chem.MolToSmiles(molecule, canonical=True)


def build_input_component_vocab(data):
    """Build first-four-component vocabularies from the provided input table."""
    columns = ['IL_SMILE', 'HL_SMILE', 'Chol_SMILE', 'PEG_SMILE']
    vocabularies = []
    for column in columns:
        keys = sorted({canonical_component_smiles(value) for value in data[column]})
        # Reserve a deterministic unknown/placeholder item so prediction-time
        # malformed components do not make an embedding lookup invalid.
        if '[Fr]' not in keys:
            keys.insert(0, '[Fr]')
        vocabularies.append({key: index for index, key in enumerate(keys)})
    return vocabularies


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
                     sample_index=None, component_vocabularies=None):
    global _CURRENT_MORDRED_FEATURE, _CURRENT_SAMPLE_UID, _CURRENT_COMPONENT_VOCAB_ID
    _CURRENT_SAMPLE_UID = torch.tensor([int(sample_index)], dtype=torch.long) if sample_index is not None else None
    for num, smiles_item_each in enumerate(smiles_item):
        # print('serial-num= ', num)
        try:
            if num < 4 and component_vocabularies is not None:
                vocabulary = component_vocabularies[num]
                vocab_id = vocabulary.get(canonical_component_smiles(smiles_item_each), 0)
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
            if cfg.use_component_aux_features:
                aux_feat = torch.from_numpy(
                    molecular_aux_features(mol_)
                ).view(1, -1)
            else:
                aux_feat = torch.zeros((1, MOLECULAR_AUX_DIM), dtype=torch.float32)
            _CURRENT_MORDRED_FEATURE = torch.from_numpy(mordred_feature_vector(
                smiles_item_each, cfg.use_mordred_features,
                cfg.mordred_feature_path, cfg.mordred_feature_dim,
            )).view(1, -1)

            if property_num == 4:
                # y = torch.Tensor([label_pair_list[0]])
                # y1 = torch.Tensor([label_pair_list[1]])
                # y2 = torch.Tensor([label_pair_list[2]])
                # y3 = torch.Tensor([label_pair_list[3]])
                y = torch.Tensor([label_pair_list[0]/100])
                y1 = torch.Tensor([label_pair_list[1]/100])
                y2 = torch.Tensor([label_pair_list[2]/100])
                y3 = torch.Tensor([label_pair_list[3]/100])
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
                            or int(ratio_pair_list[num]) == 0:
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
        component_vocabularies = build_input_component_vocab(data)
        cfg.component_vocab_sizes = [len(vocabulary) for vocabulary in component_vocabularies]
        cfg.component_vocab_source = os.path.abspath(self.raw_paths[0])
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
            sample_index_list = list(item['_stage3_source_index'])

            data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5 = [],[],[],[],[]

            if key == 0 :
                for idx in range(idx_sum):
                    if (idx %2000) == 0 :
                        print('idx',idx)
                    smi_pair_list, label_pair_list, ratio_pair_list = smiles_list[idx], label_list[idx], ratio_list[idx]
                    data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5 = smiles_to_data_5(smi_pair_list, label_pair_list,ratio_pair_list,
                                                              data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5,property_num,property_name,
                                                              sample_index=sample_index_list[idx], component_vocabularies=component_vocabularies)
            else:
                for idx in range(idx_sum):
                    if (idx %2000) == 0 :
                        print('idx',idx)
                    smi_pair_list, label_pair_list, ratio_pair_list = smiles_list[idx], label_list[idx], ratio_list[idx]
                    data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5 = smiles_to_data_5(smi_pair_list, label_pair_list,ratio_pair_list,
                                                              data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5,property_num,property_name,
                                                              sample_index=sample_index_list[idx], component_vocabularies=component_vocabularies)


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

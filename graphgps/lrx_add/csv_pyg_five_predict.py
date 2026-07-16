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

import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
from scipy.sparse import coo_matrix
import os
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import os.path as osp
from torch_geometric.data import InMemoryDataset
###from ogb.utils import smiles2graph  original version
from ogb.utils.torch_util import replace_numpy_with_torchtensor
import random
from graph_feature import smiles2graph
# picture

from torch_geometric.utils.convert import to_networkx
import networkx as nx
from torch_geometric.graphgym.config import cfg
# from ScaffoldSplitter import ScaffoldSplitter,RandomScaffoldSplitter
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions


def similar_(mol):
    # target_smi = 'N#Cc1ccccc1C#N'
    # two N#Cc1ccccc1C#N
    # one C1=CC=CC=C1C#C
    target_smi = 'O=C1C=CC(=O)N1'  # MI
    # target_smi = 'O=C1C=CC(=O)N1CN2C(=O)C=CC2=O' # BMI
    target_mol = Chem.MolFromSmiles(target_smi)
    # canonical_smi = Chem.MolToSmiles(target_mol)
    target_fp_RDKFingerprint = Chem.RDKFingerprint(target_mol)
    fp_RDKFingerprint = Chem.RDKFingerprint(mol)
    # print('RDKFingerprint', DataStructs.FingerprintSimilarity(fp_RDKFingerprint, target_fp_RDKFingerprint))
    similar_value = DataStructs.FingerprintSimilarity(fp_RDKFingerprint, target_fp_RDKFingerprint)
    return similar_value


def addCyclicConnection(mol):
    stars = []
    nbs = []
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
        print('bond already exists, skipping...')
    if (stars[0] > stars[1]):
        edmol.RemoveAtom(stars[0])
        edmol.RemoveAtom(stars[1])
    else:
        edmol.RemoveAtom(stars[1])
        edmol.RemoveAtom(stars[0])
    return edmol.GetMol()

def smiles_to_data_5(smiles_item, label_pair_list,ratio_pair_list,
                     data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5,idx_sum,property_num,property_name):

    for num, smiles_item_each in enumerate(smiles_item):
        # print('serial-num= ', num)
        try:
            if str(smiles_item_each) != 'nan':
                mol_ = Chem.MolFromSmiles(smiles_item_each)
                graph = smiles2graph(mol_)
                mask_ =False
            else:
                smiles_item_each = '[Fr]'
                mol_ = Chem.MolFromSmiles(smiles_item_each)
                graph = smiles2graph(mol_)
                mask_ = True
            # print(smiles_item)
            # graph = smiles2graph(Chem.MolFromSmiles(smiles_item))
            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            if property_num == 6:
                y = torch.Tensor(label_pair_list)
            elif property_num ==1:
                if property_name == 0:
                    y = torch.Tensor([label_pair_list[0]])
                elif property_name == 1:
                    y = torch.Tensor([label_pair_list[1]])
                elif property_name == 2:
                    y = torch.Tensor([label_pair_list[2]])
                elif property_name == 3:
                    y = torch.Tensor([label_pair_list[3]])
                elif property_name == 4:
                    y = torch.Tensor([label_pair_list[4]])
                elif property_name == 5:
                    y = torch.Tensor([label_pair_list[5]])

            # after add 2023.02.03
            edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            edge_attr = torch.from_numpy(graph['edge_feat'].flatten()).to(torch.long)

            if num == 0:
                data_1 = Data(x, edge_index, edge_attr, y, ratio=ratio_pair_list[num],mask=mask_, sum=idx_sum)
                data_sum_1.append(data_1)
            elif num == 1:
                data_2= Data(x, edge_index, edge_attr, y, ratio=ratio_pair_list[num],mask=mask_)
                data_sum_2.append(data_2)
            elif num == 2:
                data_3 = Data(x, edge_index, edge_attr, y, ratio=ratio_pair_list[num],mask=mask_)
                data_sum_3.append(data_3)
            elif num == 3:
                data_4 = Data(x, edge_index, edge_attr, y, ratio=ratio_pair_list[num],mask=mask_)
                data_sum_4.append(data_4)
            elif num == 4:
                if str(smiles_item_each) == 'nan' or str(ratio_pair_list[num]) == 'nan':
                    data_5 = Data(x, edge_index, edge_attr, y, ratio=0.0, mask=mask_)
                    data_sum_5.append(data_5)
                else:
                    data_5 = Data(x, edge_index, edge_attr, y, ratio=ratio_pair_list[num], mask=mask_)
                    data_sum_5.append(data_5)

        except:
            print('graph generate error:', num, smiles_item_each,smiles_item, y)
            # break

    return data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5


class LRX_five_predict(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,subset: bool = False,
        split: str = 'train',pre_filter= None):
        self.subset = subset
        # assert split in ['train', 'val', 'test']
        # assert split in ['train', 'val', 'test','train_j', 'val_j','test_j']
        assert split in ['train','val', 'test',
                         'train_2', 'val_2','test_2',
                         'train_3','val_3', 'test_3',
                         'train_4', 'val_4','test_4',
                         'train_5','val_5','test_5']
        super(LRX_five_predict, self).__init__(root, transform, pre_transform,pre_filter)
        self.data, self.slices = torch.load(osp.join(self.processed_dir, f'{split}.pt'))

    @property
    def raw_file_names(self) :
        return cfg.read_csv

    @property
    def processed_dir(self) -> str:
        name = 'subset' if self.subset else 'full'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self) :
        return ['train.pt', 'val.pt','test.pt',
                'train_2.pt', 'val_2.pt','test_2.pt',
                'train_3.pt', 'val_3.pt', 'test_3.pt',
                'train_4.pt', 'val_4.pt','test_4.pt',
                'train_5.pt','val_5.pt','test_5.pt']
        # return ['train.pt', 'val.pt', 'test.pt','train_j.pt', 'val_j.pt','test_j.pt']

    def download(self):
        pass

    def process(self):
        ####
        property_num = cfg.property_num
        property_name = cfg.property_serial
        ####
        data = pd.read_csv(self.raw_paths[0])

        train_data = data[:cfg.train.batch_size]
        valid_data = data
        test_data = data

        list_splite =[train_data,valid_data,test_data]
        for key,item in enumerate(list_splite):
            idx_sum = len(item)
            print('key =', key, 'len(item)',idx_sum)

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
            y1_list = item['EE_before'].tolist()
            y2_list = item['EE_after'].tolist()
            y3_list = item['Aerosolization_Efficiency'].tolist()
            y4_list = item['mRNA_Recovery_Efficiency'].tolist()
            y5_list = item['Norm_before'].tolist()
            y6_list = item['Norm_after'].tolist()

            smiles_list = [list(pair_smi) for pair_smi in zip(molecule_1, molecule_2,molecule_3,molecule_4,molecule_5)]
            ratio_list = [list(pair_rate) for pair_rate in zip(rate_1, rate_2, rate_3, rate_4, rate_5)]
            label_list = [list(pair_y) for pair_y in zip(y1_list, y2_list, y3_list, y4_list, y5_list,y6_list)]


            data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5 = [],[],[],[],[]

            if key == 0 :
                for idx in range(idx_sum):
                    if (idx %2000) == 0 :
                        print('idx',idx)
                    smi_pair_list, label_pair_list, ratio_pair_list = smiles_list[idx], label_list[idx], ratio_list[idx]
                    data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5 = smiles_to_data_5(smi_pair_list, label_pair_list,ratio_pair_list,
                                                              data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5,idx_sum,property_num,property_name)
                if len(data_sum_1) % cfg.train.batch_size != 0:
                    gap_value = int(cfg.train.batch_size) - (len(data_sum_1) % cfg.train.batch_size)
                    data_sum_1_add, data_sum_2_add, data_sum_3_add, data_sum_4_add, data_sum_5_add = [], [], [], [], []
                    for item in range(gap_value):
                        data_sum_1_add, data_sum_2_add, data_sum_3_add, data_sum_4_add, data_sum_5_add = smiles_to_data_5(smi_pair_list,
                                                                                                      label_pair_list,
                                                                                                      ratio_pair_list,
                                                                                                      data_sum_1_add, data_sum_2_add, data_sum_3_add, data_sum_4_add, data_sum_5_add,idx_sum,property_num,property_name)
                    for serial_add,add_item in enumerate(data_sum_1_add):
                        data_sum_1.append(add_item)
                        data_sum_2.append(data_sum_2_add[serial_add])
                        data_sum_3.append(data_sum_3_add[serial_add])
                        data_sum_4.append(data_sum_4_add[serial_add])
                        data_sum_5.append(data_sum_5_add[serial_add])
                    if len(data_sum_1) % cfg.train.batch_size !=0 :
                        print('---- error ----')


            else:
                for idx in range(idx_sum):
                    if (idx %2000) == 0 :
                        print('idx',idx)
                    smi_pair_list, label_pair_list, ratio_pair_list = smiles_list[idx], label_list[idx], ratio_list[idx]
                    data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5 = smiles_to_data_5(smi_pair_list, label_pair_list,ratio_pair_list,
                                                              data_sum_1, data_sum_2,data_sum_3, data_sum_4,data_sum_5,idx_sum,property_num,property_name)
                if len(data_sum_1) % cfg.train.batch_size != 0:
                    gap_value = int(cfg.train.batch_size) - (len(data_sum_1) % cfg.train.batch_size)
                    data_sum_1_add, data_sum_2_add, data_sum_3_add, data_sum_4_add, data_sum_5_add = [], [], [], [], []
                    for item in range(gap_value):
                        data_sum_1_add, data_sum_2_add, data_sum_3_add, data_sum_4_add, data_sum_5_add = smiles_to_data_5(smi_pair_list,
                                                                                                      label_pair_list,
                                                                                                      ratio_pair_list,
                                                                                                      data_sum_1_add, data_sum_2_add, data_sum_3_add, data_sum_4_add, data_sum_5_add,idx_sum,property_num,property_name)
                    for serial_add,add_item in enumerate(data_sum_1_add):
                        data_sum_1.append(add_item)
                        data_sum_2.append(data_sum_2_add[serial_add])
                        data_sum_3.append(data_sum_3_add[serial_add])
                        data_sum_4.append(data_sum_4_add[serial_add])
                        data_sum_5.append(data_sum_5_add[serial_add])
                    if len(data_sum_1) % cfg.train.batch_size !=0 :
                        print('---- error ----')


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

            print('------------ending--------------')

        return data_sum_1, data_sum_2, data_sum_3, data_sum_4, data_sum_5




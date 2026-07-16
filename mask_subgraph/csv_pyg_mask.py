"""
@Name:  csv_pyg_mask.py
@Auth:  rongxing
@Date:  2023/2/9-上午10:23
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

add masked subgraph module
"""

import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
from scipy.sparse import coo_matrix
import os
import rdkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import os.path as osp
from torch_geometric.data import InMemoryDataset
###from ogb.utils import smiles2graph  original version
from ogb.utils.torch_util import replace_numpy_with_torchtensor
import random
from graph_feature import smiles2graph
from mask_subgraph.mask_subgraph import mask_subgraph
import logging
from torch_geometric.graphgym.config import cfg

class LRX_mask_subgraph(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,subset: bool = False,
        split: str = 'train',pre_filter= None):
        self.subset = subset
        assert split in ['train', 'val', 'test','train_j', 'val_j','test_j']
        super(LRX_mask_subgraph, self).__init__(root, transform, pre_transform,pre_filter)
        self.data, self.slices = torch.load(osp.join(self.processed_dir, f'{split}.pt'))


    @property
    def raw_file_names(self) :
        # return './mask/pubchem-10m-clean_element_short.csv'
        return './test/pubchem-10m-clean_element_test.csv'
        # return 'pubchem-10m-clean_element_short.csv'
        # return '250k_rndm_zinc_drugs_clean_3.csv'
        # return 'clean_element_canon_1995+PN+7841_reset.csv'

    @property
    def processed_dir(self) -> str:
        name = 'subset' if self.subset else 'full'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self) :

        return ['train.pt', 'val.pt','test.pt','train_j.pt', 'val_j.pt','test_j.pt']
        # return ['train.pt', 'val.pt', 'test.pt']


    def download(self):
        pass
    # def get(self, idx):

    def process(self):
        data = (pd.read_csv(self.raw_paths[0])) #[:100]
        print('len data',len(data))
        data,unusedata = train_test_split(data, train_size=0.1, test_size=0.1, random_state = int(cfg.seed))
        print('len data', len(data))
        # train_data, test_data = train_test_split(data, train_size=0.9, test_size=0.1, random_state=int(cfg.seed))
        # valid_data, test_data = train_test_split(test_data, train_size=0.99, test_size=0.01, random_state=int(cfg.seed))
        #
        # list_splite =[train_data,valid_data,test_data]
        list_splite = [data, data, data]
        # logging.info(f"[*] train_data {len(train_data)}: test_data={len(test_data)}, "
        #              f"valid_data={len(valid_data)}")
        for key,item in enumerate(list_splite):
            item.reset_index(drop=True)
            idx_sum = len(item)
            print('key =', key, 'len(item)',idx_sum)
            # logging.info(f"len(smiles_list): {len(smiles_list)}")
            smiles_list = list(item['canonical smiles'])
            Tm_list = list(item['Tm_mean'])
            print(len(smiles_list),len(Tm_list))

            data_sum = []
            data_sum_i = []
            data_sum_j = []
            repeat_number = 0 #10
            repeat_number_max = repeat_number + 20
            max_repeat_condition = True #True  #是否达到恒定数量
            repeat_condition = True #False #是否有重复
            if key == 0:
                # smiles_total_list =[]
                for idx in range(idx_sum):
                    if (idx %2000) == 0 :
                        print('idx',idx)
                    smiles, Tm = smiles_list[idx], Tm_list[idx]
                    smiles_item_list = []
                    Tm_repeat_list = []
                    smiles_item_list.append(smiles)
                    count =0
                    # print(idx,smiles)
                    #add repeat smiles
                    while len(smiles_item_list) <= repeat_number:
                        mol = Chem.MolFromSmiles(smiles)
                        # print('len(smiles_item_list)',len(smiles_item_list))
                        if mol:
                            Chem.Kekulize(mol)
                            repeat_random_smiles = rdkit.Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False,
                                                          kekuleSmiles=True)
                        count += 1
                        if repeat_random_smiles != '':
                            smiles_item_list.append(repeat_random_smiles)
                        if repeat_condition == False:
                            smiles_item_list = list(set(smiles_item_list))
                        elif repeat_condition == True:
                            if max_repeat_condition == False:
                                smiles_item_list = list(smiles_item_list)
                            elif max_repeat_condition == True:
                                smiles_item_list = list(set(smiles_item_list))
                                if count > repeat_number_max:
                                    add_num = repeat_number - int(len(smiles_item_list)) + 1
                                    for item in range(add_num):
                                        smiles_item_list.append(smiles)
                                    # print(len(smiles_item_list),add_num)
                    for serial,smiles_item in enumerate(smiles_item_list):

                        data_i, data_j = mask_subgraph(smiles_item,Tm)
                        data_sum_i.append(data_i)
                        data_sum_j.append(data_j)
            else:
                for idx in range(idx_sum):
                    smiles, Tm = smiles_list[idx], Tm_list[idx]
                    data_i, data_j = mask_subgraph(smiles,Tm)
                    data_sum_i.append(data_i)
                    data_sum_j.append(data_j)
                    ####  Data(x, edge_index, edge_attr, y)

            if key == 0:
                # A_csv_name = ['canonical smiles']
                # A_csv = pd.DataFrame(columns=A_csv_name, data=smiles_total_list)
                # A_csv.to_csv('./smiles.csv')
                print('train :len(data_sum)',len(data_sum_i),len(data_sum_j))
                torch.save(self.collate(data_sum_i), os.path.join(self.processed_dir, f'train.pt'))
                torch.save(self.collate(data_sum_j), os.path.join(self.processed_dir, f'train_j.pt'))
            elif key == 1 :
                print('val :len(data_sum)', len(data_sum_i),len(data_sum_j))
                torch.save(self.collate(data_sum_i), os.path.join(self.processed_dir, f'val.pt'))
                torch.save(self.collate(data_sum_j), os.path.join(self.processed_dir, f'val_j.pt'))
            elif key == 2:
                print('test :len(data_sum)', len(data_sum_i),len(data_sum_j))
                torch.save(self.collate(data_sum_i), os.path.join(self.processed_dir, f'test.pt'))
                torch.save(self.collate(data_sum_j), os.path.join(self.processed_dir, f'test_j.pt'))
            # print('------------ending--------------')

        return data_sum_i,data_sum_j

    def randomize_smiles(mol):
        '''Returns a random (dearomatized) SMILES given an rdkit mol object of a molecule.

        Parameters:
        mol (rdkit.Chem.rdchem.Mol) :  RdKit mol object (None if invalid smile string smi)

        Returns:
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object  (None if invalid smile string smi)
        '''
        if not mol:
            return None

        Chem.Kekulize(mol)
        return rdkit.Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False, kekuleSmiles=True)


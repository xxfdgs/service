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

    # target_fp_GenMACCSKeys = MACCSkeys.GenMACCSKeys(target_mol)
    # fp_GenMACCSKeys = MACCSkeys.GenMACCSKeys(mol)
    # # print('GenMACCSKeys', DataStructs.FingerprintSimilarity(fp_GenMACCSKeys, target_fp_GenMACCSKeys))
    # similar_value = DataStructs.FingerprintSimilarity(fp_GenMACCSKeys, target_fp_GenMACCSKeys)
    # return similar_value

    # target_fp_GetAtomPairFingerprint = Pairs.GetAtomPairFingerprint(target_mol)
    # fp_GetTopologicalTorsionFingerprintAsIntVect = Pairs.GetAtomPairFingerprint(mol)
    # # print('GetAtomPairFingerprint',
    # #       DataStructs.DiceSimilarity(fp_GetAtomPairFingerprint, target_fp_GetAtomPairFingerprint))
    # similar_value = DataStructs.DiceSimilarity(target_fp_GetAtomPairFingerprint, fp_GetTopologicalTorsionFingerprintAsIntVect)
    # return similar_value

    # target_fp_GetTopologicalTorsionFingerprintAsIntVect = Torsions.GetTopologicalTorsionFingerprintAsIntVect(
    #     target_mol)
    # fp_GetTopologicalTorsionFingerprintAsIntVect = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
    # # print('GetTopologicalTorsionFingerprintAsIntVect',
    # #       DataStructs.DiceSimilarity(fp_GetTopologicalTorsionFingerprintAsIntVect,
    # #                                  target_fp_GetTopologicalTorsionFingerprintAsIntVect))
    # similar_value = DataStructs.DiceSimilarity(fp_GetTopologicalTorsionFingerprintAsIntVect,
    #                            target_fp_GetTopologicalTorsionFingerprintAsIntVect)
    # return similar_value

    # target_fp_GetMACCSKeysFingerprint = AllChem.GetMACCSKeysFingerprint(target_mol)
    # fp_GetMACCSKeysFingerprint = AllChem.GetMACCSKeysFingerprint(mol)
    # # print('GetMACCSKeysFingerprint',
    # #       DataStructs.FingerprintSimilarity(fp_GetMACCSKeysFingerprint, target_fp_GetMACCSKeysFingerprint))
    # similar_value = DataStructs.FingerprintSimilarity(fp_GetMACCSKeysFingerprint, target_fp_GetMACCSKeysFingerprint)
    # return similar_value


    # target_fp_GetMorganFingerprint = AllChem.GetMorganFingerprint(target_mol, 2)
    # fp_GetMorganFingerprint = AllChem.GetMorganFingerprint(mol, 2)
    # # print('GetMorganFingerprint',
    # #       DataStructs.DiceSimilarity(fp_GetMorganFingerprint, target_fp_GetMorganFingerprint))
    # similar_value = DataStructs.DiceSimilarity(fp_GetMorganFingerprint, target_fp_GetMorganFingerprint)
    # return similar_value

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

def smiles_to_data_rate(serial,smiles_item,Tm,data_sum_i,data_sum_j, input_ratio,ad1):
    data_sum_i_old = data_sum_i
    data_sum_j_old = data_sum_j
    data_sum_i_new , data_sum_j_new = [],[]
    for num, smiles_item_each in enumerate(smiles_item):
        # print('serial-num= ', num)
        try:
            ### */no *
            if smiles_item_each.count('*') != 0:
                mol_ = addCyclicConnection(Chem.MolFromSmiles(smiles_item_each))
                graph = smiles2graph(mol_)
            elif smiles_item_each.count('*') == 0:
                mol_ = Chem.MolFromSmiles(smiles_item_each)
                graph = smiles2graph(mol_)
            # print(smiles_item)
            # graph = smiles2graph(Chem.MolFromSmiles(smiles_item))
            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            # y = torch.Tensor([Tm])
            # add error in the range of 5 temperature. 2023.04.10
            # if serial == 0:
            y = torch.Tensor([float(Tm)])

            # after add 2023.02.03
            edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            edge_attr = torch.from_numpy(graph['edge_feat'].flatten()).to(torch.long)

            ### lrx add similar value
            mol = Chem.MolFromSmiles(smiles_item_each)
            similarity = round(similar_(mol),3)
            # similarity = 1 / abs(y)
            mnw = 0
            ### A+B =1
            input_ratio_list = input_ratio.split('/')
            if ad1 == False:
                if input_ratio_list[0] >= input_ratio_list[1]:
                    ratio_list = [1.0, round((float(input_ratio_list[1]) / float(input_ratio_list[0])), 2)]
                elif input_ratio_list[0] < input_ratio_list[1]:
                    ratio_list = [round((float(input_ratio_list[0]) / float(input_ratio_list[1])), 2), 1.0]
                if num == 0:
                    ratio_0 = round(float(input_ratio_list[0]),3)
                elif num == 1:
                    ratio_1 = round((1 - round(float(input_ratio_list[0]),3)),3)
            # data_ = Data(x, edge_index, edge_attr, y, similarity=similarity, mnw=mnw, ratio=ratio)
            # data_ = Data(x, edge_index, edge_attr, y, ratio=ratio, similarity=similarity)
            else:
                if num == 0:
                    # ratio_0 = 1.0
                    ratio_0 = round(float(input_ratio_list[0]), 3)
                elif num == 1:
                    # ratio_1 = 0.0
                    ratio_1 = round((1 - round(float(input_ratio_list[0]), 3)), 3)
            if num == 0:
                data_ = Data(x, edge_index, edge_attr, y, ratio=ratio_0, similarity=similarity)
                data_sum_i_new.append(data_)
            else:
                data_ = Data(x, edge_index, edge_attr, y, ratio=ratio_1, similarity=similarity)
                data_sum_j_new.append(data_)


        except:
            print('graph generate error:', serial, smiles_item_each,smiles_item, y)
            # break
    if len(data_sum_i_new) !=0 and  len(data_sum_j_new) !=0 :
        data_sum_i.append(data_sum_i_new[0])
        data_sum_j.append(data_sum_j_new[0])
        return data_sum_i, data_sum_j
    else:
        print('error ',len(data_sum_i),len(data_sum_j))
        print('error ', len(data_sum_i_old), len(data_sum_j_old))
        return data_sum_i_old,data_sum_j_old

def smiles_to_data_no_rate(serial,smiles_item,Tm,data_sum_i,data_sum_j, input_ratio,ad1):
    data_sum_i_old = data_sum_i
    data_sum_j_old = data_sum_j
    data_sum_i_new , data_sum_j_new = [],[]
    for num, smiles_item_each in enumerate(smiles_item):
        # print('serial-num= ', num)
        try:
            ### */no *
            if smiles_item_each.count('*') != 0:
                mol_ = addCyclicConnection(Chem.MolFromSmiles(smiles_item_each))
                graph = smiles2graph(mol_)
            elif smiles_item_each.count('*') == 0:
                mol_ = Chem.MolFromSmiles(smiles_item_each)
                graph = smiles2graph(mol_)
            # print(smiles_item)
            # graph = smiles2graph(Chem.MolFromSmiles(smiles_item))
            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            # y = torch.Tensor([Tm])
            # add error in the range of 5 temperature. 2023.04.10
            if serial == 0:
                y = torch.Tensor([float(Tm)])

            # after add 2023.02.03
            edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            edge_attr = torch.from_numpy(graph['edge_feat'].flatten()).to(torch.long)

            ### lrx add similar value
            mol = Chem.MolFromSmiles(smiles_item_each)
            similarity = similar_(mol)
            # similarity = 1 / abs(y)
            mnw = 0
            ratio = 0.5
            data_ = Data(x, edge_index, edge_attr, y, ratio=ratio,similarity=similarity)
            if num == 0:
                data_sum_i_new.append(data_)
            else:
                data_sum_j_new.append(data_)



        except:
            print('graph generate error:', serial, smiles_item_each)
            # break
    if len(data_sum_i_new) !=0 and  len(data_sum_j_new) !=0 :
        data_sum_i.append(data_sum_i_new[0])
        data_sum_j.append(data_sum_j_new[0])
        return data_sum_i, data_sum_j
    else:
        print('error ',len(data_sum_i),len(data_sum_j))
        print('error ', len(data_sum_i_old), len(data_sum_j_old))
        return data_sum_i_old,data_sum_j_old

class LRX_mask_classify(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,subset: bool = False,
        split: str = 'train',pre_filter= None):
        self.subset = subset
        assert split in ['train', 'val', 'test','train_j', 'val_j','test_j']
        super(LRX_mask_classify, self).__init__(root, transform, pre_transform,pre_filter)
        self.data, self.slices = torch.load(osp.join(self.processed_dir, f'{split}.pt'))
        print('-ending--')


    @property
    def raw_file_names(self) :
        return './mask/pubchem-10m-clean_element_short.csv'
        # return './test/co_DP_rate_mix_max_tg_test.csv'

    @property
    def processed_dir(self) -> str:
        name = 'subset' if self.subset else 'full'
        # print('processed_dir',osp.join(self.root, name, 'processed'))
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self) :
        # print('processed_file_names')
        return ['train.pt', 'val.pt', 'test.pt','train_j.pt', 'val_j.pt','test_j.pt']

    def download(self):
        pass
    # def get(self, idx):

    def process(self):
        ####
        rate_condition = cfg.data_rate # True
        ad1 = cfg.data_rate_type  ### ad1 == True 代表 a+b=1  否则 代表 1:x
        ####
        data = pd.read_csv(self.raw_paths[0])
        # data_norate = pd.read_csv(self.raw_paths[1])

        train_data, test_data = train_test_split(data, train_size=0.9, test_size=0.1, random_state = int(cfg.seed))
        train_data, valid_data = train_test_split(train_data, train_size=0.9, test_size=0.1, random_state = int(cfg.seed))
        # print(len(train_data),len(test_data),len(valid_data),len(data_norate))
        # train_data = pd.concat([train_data, data_norate], axis=0, ignore_index=True)
        # print(len(train_data), len(test_data), len(valid_data), len(data_norate))

        list_splite = [train_data, valid_data, test_data]
        for key,item in enumerate(list_splite):
            item.reset_index(drop=True)
            idx_sum = len(item)
            print('key =', key, 'len(item)',idx_sum)
            ### polymer type
            Tm_list = list(item['Tg_mean'])
            molecule_1 = list(item['molecule 1'])
            molecule_2 = list(item['molecule 2'])
            smiles_list = [list(pair) for pair in zip(molecule_1, molecule_2)]#zip(molecule_1,molecule_2)
            # mnw_list  = list(item['Mn'])
            ratio_list = list(item['rate'])
            ###

            print(len(smiles_list),len(Tm_list))
            data_sum = []
            data_sum_i, data_sum_j = [],[]
            if key == 0 :
                for idx in range(idx_sum):
                    if (idx %2000) == 0 :
                        print('idx',idx)
                    smiles, Tm, ratio = smiles_list[idx], Tm_list[idx],ratio_list[idx]
                    smiles_item_list = []
                    smiles_item_list.append(smiles)

                    for serial, smiles_item in enumerate(smiles_item_list):
                        # print(serial,smiles_item)
                        if rate_condition == True:
                            data_sum_i, data_sum_j = smiles_to_data_rate(idx, smiles_item, Tm, data_sum_i, data_sum_j, ratio,ad1)
                        else:
                            data_sum_i, data_sum_j = smiles_to_data_no_rate(serial, smiles_item, Tm, data_sum_i, data_sum_j, ratio,ad1)

            else:
                for idx in range(idx_sum):
                    # smiles, Tm = smiles_list[idx], Tm_list[idx]
                    smiles, Tm, ratio = smiles_list[idx], Tm_list[idx], ratio_list[idx]
                    smiles_item_list = []
                    # Tm_repeat_list = []
                    smiles_item_list.append(smiles)

                    for serial, smiles_item in enumerate(smiles_item_list):
                        if rate_condition == True:
                            data_sum_i, data_sum_j = smiles_to_data_rate(serial, smiles_item, Tm, data_sum_i, data_sum_j, ratio,ad1)
                        else:
                            data_sum_i, data_sum_j = smiles_to_data_no_rate(serial, smiles_item, Tm, data_sum_i, data_sum_j, ratio,ad1)


            if key == 0:
                ##打乱顺序
                # random.shuffle(data_sum)
                print('train :len(data_sum_1),len(data_sum_2)', len(data_sum_i), len(data_sum_j))
                torch.save(self.collate(data_sum_i), os.path.join(self.processed_dir, f'train.pt'))
                torch.save(self.collate(data_sum_j), os.path.join(self.processed_dir, f'train_j.pt'))
            elif key == 1 :
                print('val :len(data_sum_1),len(data_sum_2)', len(data_sum_i), len(data_sum_j))
                torch.save(self.collate(data_sum_i), os.path.join(self.processed_dir, f'val.pt'))
                torch.save(self.collate(data_sum_j), os.path.join(self.processed_dir, f'val_j.pt'))
            elif key == 2:
                print('test :len(data_sum_1),len(data_sum_2)', len(data_sum_i), len(data_sum_j))
                torch.save(self.collate(data_sum_i), os.path.join(self.processed_dir, f'test.pt'))
                torch.save(self.collate(data_sum_j), os.path.join(self.processed_dir, f'test_j.pt'))
            print('------------ending--------------')

        return data_sum_i, data_sum_j
"""
@Name:  mask_subgraph.py
@Auth:  rongxing
@Date:  2023/2/9-下午2:20
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME}
def remove_subgraph
def mask_subgraph
"""
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy
import networkx as nx

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
# import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

# def save_picture(data,serial):
#     # save picture
#
#     fig1 = plt.figure(1)
#     graph = to_networkx(data)
#     pos = nx.spring_layout(graph)
#     nx.draw_networkx_nodes(graph, pos)
#     nx.draw_networkx_edges(graph, pos)
#     nx.draw_networkx_labels(graph, pos)
#     # plt.show()
#     plt.savefig('/home/lrx/dataset/code/GraphGPS-main/results/result_' + str(serial) + '.png')
#     plt.clf()

def remove_subgraph(Graph, center, percent=0.2):
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes) * percent))
    removed = []
    temp = [center]

    while len(removed) < num:
        neighbors = []
        if len(temp) < 1:
            break

        for n in temp:
            neighbors.extend([i for i in G.neighbors(n) if i not in temp])  ## g.neighbors(1) 所有与1这个点相连的点的信息以列表的形式返回
        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break

        temp = list(set(neighbors))
    return G, removed


def mask_subgraph(smiles_item,Tm):
    mol = Chem.MolFromSmiles(smiles_item)
    mol = Chem.AddHs(mol)

    N = mol.GetNumAtoms()
    M = mol.GetNumBonds()
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    #########################
    # Get the molecule info #
    #########################
    type_idx = []
    chirality_idx = []
    atomic_number = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

    ####################
    # Subgraph Masking #
    ####################

    # Construct the original molecular graph from edges (bonds)
    edges = []
    for bond in bonds:
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    molGraph = nx.Graph(edges)

    # Get the graph for i and j after removing subgraphs
    start_i, start_j = random.sample(list(range(N)), 2)
    percent_i, percent_j = random.uniform(0, 0.2), random.uniform(0, 0.2)
    G_i, removed_i = remove_subgraph(molGraph, start_i, percent=percent_i)
    G_j, removed_j = remove_subgraph(molGraph, start_j, percent=percent_j)

    atom_remain_indices_i = [i for i in range(N) if i not in removed_i]
    atom_remain_indices_j = [i for i in range(N) if i not in removed_j]

    # Only consider bond still exist after removing subgraph
    row_i, col_i, row_j, col_j = [], [], [], []
    edge_feat_i, edge_feat_j = [], []
    G_i_edges = list(G_i.edges)
    G_j_edges = list(G_j.edges)

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = [
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ]
        if (start, end) in G_i_edges or (end, start) in G_i_edges:
            row_i += [start, end]
            col_i += [end, start]
            edge_feat_i.append(feature)
            edge_feat_i.append(feature)
        if (start, end) in G_j_edges or (end, start) in G_j_edges:
            row_j += [start, end]
            col_j += [end, start]
            edge_feat_j.append(feature)
            edge_feat_j.append(feature)

    edge_index_i = torch.tensor([row_i, col_i], dtype=torch.long)
    edge_attr_i = torch.tensor(np.array(edge_feat_i), dtype=torch.long)
    edge_index_j = torch.tensor([row_j, col_j], dtype=torch.long)
    edge_attr_j = torch.tensor(np.array(edge_feat_j), dtype=torch.long)

    ############################
    # Random Atom/Edge Masking #
    ############################

    num_mask_nodes_i = max([0, math.floor(0.25 * N) - len(removed_i)])
    num_mask_edges_i = max([0, edge_attr_i.size(0) // 2 - math.ceil(0.75 * M)])
    num_mask_nodes_j = max([0, math.floor(0.25 * N) - len(removed_j)])
    num_mask_edges_j = max([0, edge_attr_j.size(0) // 2 - math.ceil(0.75 * M)])
    mask_nodes_i = random.sample(atom_remain_indices_i, num_mask_nodes_i)
    mask_nodes_j = random.sample(atom_remain_indices_j, num_mask_nodes_j)
    mask_edges_i_single = random.sample(list(range(edge_attr_i.size(0) // 2)), num_mask_edges_i)
    mask_edges_j_single = random.sample(list(range(edge_attr_j.size(0) // 2)), num_mask_edges_j)
    mask_edges_i = [2 * i for i in mask_edges_i_single] + [2 * i + 1 for i in mask_edges_i_single]
    mask_edges_j = [2 * i for i in mask_edges_j_single] + [2 * i + 1 for i in mask_edges_j_single]

    x_i = deepcopy(x)
    for atom_idx in range(N):
        if (atom_idx in mask_nodes_i) or (atom_idx in removed_i):
            x_i[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
    edge_index_final_i = torch.zeros((2, edge_attr_i.size(0) - 2 * num_mask_edges_i), dtype=torch.long)
    edge_attr_final_i = torch.zeros((edge_attr_i.size(0) - 2 * num_mask_edges_i, 2), dtype=torch.long)
    count = 0
    for bond_idx in range(edge_attr_i.size(0)):
        if bond_idx not in mask_edges_i:
            edge_index_final_i[:, count] = edge_index_i[:, bond_idx]
            edge_attr_final_i[count, :] = edge_attr_i[bond_idx, :]
            count += 1
    data_i = Data(x=x_i, edge_index=edge_index_final_i, edge_attr=edge_attr_final_i)
    # save_picture(data_i,1)
    #### delete redundant column
    ### maybe don't run
    x_i_ts = deepcopy(x_i)
    x_i_np = x_i_ts.numpy()
    x_i_np_new = torch.tensor([item[0] for item in x_i_np], dtype=torch.long).view(-1, 1)
    ### must be run
    edge_attr_final_i_ts = deepcopy(edge_attr_final_i)
    edge_attr_final_i_np = edge_attr_final_i_ts.numpy()
    # edge_attr_final_i_np_new = torch.tensor([item[0] for item in edge_attr_final_i_np], dtype=torch.long).view(-1, 1)
    edge_attr_final_i_np_new = torch.tensor([item[0] for item in edge_attr_final_i_np], dtype=torch.long).flatten()

    data_i_new = Data(x=x_i_np_new, edge_index=edge_index_final_i, edge_attr=edge_attr_final_i_np_new,y = torch.Tensor([Tm]))
    # save_picture(data_i_new, 2)
    #
    x_j = deepcopy(x)
    for atom_idx in range(N):
        if (atom_idx in mask_nodes_j) or (atom_idx in removed_j):
            x_j[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
    edge_index_final_j = torch.zeros((2, edge_attr_j.size(0) - 2 * num_mask_edges_j), dtype=torch.long)
    edge_attr_final_j = torch.zeros((edge_attr_j.size(0) - 2 * num_mask_edges_j, 2), dtype=torch.long)
    count = 0
    for bond_idx in range(edge_attr_j.size(0)):
        if bond_idx not in mask_edges_j:
            edge_index_final_j[:, count] = edge_index_j[:, bond_idx]
            edge_attr_final_j[count, :] = edge_attr_j[bond_idx, :]
            count += 1
    data_j = Data(x=x_j, edge_index=edge_index_final_j, edge_attr=edge_attr_final_j)
    # save_picture(data_j, 3)
    #### delete redundant column
    x_j_ts = deepcopy(x_j)
    x_j_np = x_j_ts.numpy()
    x_j_np_new = torch.tensor([item[0] for item in x_j_np], dtype=torch.long).view(-1, 1)
    edge_attr_final_j_ts = deepcopy(edge_attr_final_j)
    edge_attr_final_j_np = edge_attr_final_j_ts.numpy()
    # edge_attr_final_j_np_new = torch.tensor([item[0] for item in edge_attr_final_j_np], dtype=torch.long).view(-1, 1)
    edge_attr_final_j_np_new = torch.tensor([item[0] for item in edge_attr_final_j_np], dtype=torch.long).flatten()
    #flatten()
    data_j_new = Data(x=x_j_np_new, edge_index=edge_index_final_j, edge_attr=edge_attr_final_j_np_new,y = torch.Tensor([Tm]))
    # save_picture(data_j_new, 4)
    #### return data_i, data_j

    # # save picture
    # fig1 = plt.figure(1)
    # graph_j = to_networkx(data_j)
    # pos_j = nx.spring_layout(graph_j)
    # nx.draw_networkx_nodes(graph_j, pos_j)
    # nx.draw_networkx_edges(graph_j, pos_j)
    # nx.draw_networkx_labels(graph_j, pos_j)
    # # plt.show()
    # plt.savefig('/home/lrx/dataset/code/GraphGPS-main/results/result_'+ str(4) +'.png')

    return data_i_new, data_j_new
# if __name__ == '__main__':
